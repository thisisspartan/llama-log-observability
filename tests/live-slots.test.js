'use strict';
// Фаза 1 (C1/C2/C5): /slots.id_task — единственный авторитетный ключ текущего task.
// Интеграционный тест: mock llama-server (/slots) + www.js в LIVE-режиме +
// два последовательных task через stdin. Доказывает, что stale /slots-ответ
// task A НЕ способен перезаписать счётчики task B.
// Запуск: node --test tests/
//
// Тайминг: waitState() — condition polling (проверка условия каждые 100 мс,
// таймаут 8 с) вместо фиксированных sleep(1200). Монитор работает в ОТДЕЛЬНОМ
// процессе (spawn), поэтому node:test mock.timers до него не дотягивается —
// condition polling является корректным эквивалентом: быстрее (не ждём лишнего)
// и детерминированнее (не зависим от того, успел ли поллинг за фиксированный сон).
// Для доказательств «не перезаписал» (значение НЕ должно измениться) условия
// недостаточно — там ждём ровно один полный цикл поллинга монитора (POLL_CYCLE:
// setInterval(pollServer, 500) + flushState 100 мс в www.js) и сверяем снимки.
const { test } = require('node:test');
const assert = require('node:assert');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');

const WWW = path.join(__dirname, '..', 'www.js');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const POLL_CYCLE = 700; // ≥ один цикл pollServer(500 мс) + flushState(100 мс)

// Скриптуемый mock llama-server: тест выставляет mock.slotPayload,
// /slots отдаёт его текущее значение (так моделируется stale HTTP-ответ).
function startMockSlots() {
  const mock = { slotPayload: [] };
  const srv = http.createServer((req, res) => {
    if (req.url === '/slots') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(mock.slotPayload));
    } else {
      res.writeHead(404);
      res.end();
    }
  });
  return new Promise(resolve => srv.listen(0, '127.0.0.1', () => {
    mock.port = srv.address().port;
    resolve({ mock, srv });
  }));
}

// Форма ответа /slots — как в to_json (llama.cpp server-context.cpp)
function slotPayload({ idTask, processing, prompt, processed, cached, decoded, nCtx = 124160 }) {
  return [{
    id: 0,
    n_ctx: nCtx,
    speculative: true,
    is_processing: processing,
    id_task: idTask,
    n_prompt_tokens: prompt,
    n_prompt_tokens_processed: processed,
    n_prompt_tokens_cache: cached,
    next_token: [{ has_next_token: false, has_new_line: false, n_remain: -1, n_decoded: decoded }]
  }];
}

async function startMonitor(mockPort) {
  const monPort = 40000 + Math.floor(Math.random() * 9000);
  let stderr = '';
  const child = spawn('node', [WWW, '--port', String(monPort), '--url', `http://127.0.0.1:${mockPort}`], {
    stdio: ['pipe', 'ignore', 'pipe'],
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' }
  });
  child.stderr.on('data', d => { stderr += d; if (stderr.length > 8000) stderr = stderr.slice(-8000); });
  // Ждём, пока монитор начнёт отвечать
  let up = false;
  for (let i = 0; i < 100 && !up; i++) {
    await sleep(100);
    try { up = (await fetch(`http://127.0.0.1:${monPort}/api/state`)).ok; } catch (_) { /* ещё стартует */ }
  }
  assert.ok(up, 'monitor is up (stderr: ' + stderr + ')');
  return { child, monPort, getStderr: () => stderr };
}

async function fetchState(monPort) {
  const res = await fetch(`http://127.0.0.1:${monPort}/api/state`);
  assert.ok(res.ok, 'GET /api/state -> ' + res.status);
  return res.json();
}

// condition polling: ждём, пока состояние монитора удовлетворит pred;
// по таймаунту — падение с последним снимком (диагностика, не тихий сон)
async function waitState(monPort, pred, label, timeoutMs = 8000) {
  const t0 = Date.now();
  for (;;) {
    const s = await fetchState(monPort);
    if (pred(s)) return s;
    if (Date.now() - t0 > timeoutMs) {
      throw new Error('waitState timeout (' + label + '): last=' +
        JSON.stringify({ n_prompt_tokens: s.n_prompt_tokens, task: s.task }));
    }
    await sleep(100);
  }
}

test('C1/C2/C5: stale /slots response of task A cannot overwrite task B', async () => {
  const { mock, srv } = await startMockSlots();
  const { child, monPort, getStderr } = await startMonitor(mock.port);
  try {
    // Фаза 0: отслеживаемого task нет (id == null) — слот даёт только
    // slot-level поля (occupancy/is_processing), task-счётчики не трогаются
    mock.slotPayload = slotPayload({ idTask: 999, processing: true, prompt: 5000, processed: 4000, cached: 1000, decoded: 300 });
    let s = await waitState(monPort, s => s.n_prompt_tokens === 5000 && s.task.id === null,
      'фаза 0: occupancy применён, task не отслеживается');
    assert.strictEqual(s.n_prompt_tokens, 5000, 'occupancy из /slots');
    assert.strictEqual(s.is_processing, true);
    assert.strictEqual(s.task.id, null);
    assert.strictEqual(s.task.kvRestoredTokens, 0, 'task-счётчики без активного task не применяются');
    assert.strictEqual(s.task.prefillTokens, 0);
    assert.strictEqual(s.task.generatedTokens, 0);

    // Фаза 1: task A (id 1) — new prompt из stdin, /slots с id_task=1
    mock.slotPayload = slotPayload({ idTask: 1, processing: true, prompt: 1000, processed: 200, cached: 800, decoded: 50 });
    child.stdin.write('slot operator(): id  0 | task 1 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000\n');
    s = await waitState(monPort, s => s.task.id === '1' && s.task.prefillTokens === 200 && s.task.generatedTokens === 50,
      'фаза 1: счётчики task A применены');
    assert.strictEqual(s.task.id, '1');
    assert.strictEqual(s.task.totalPrompt, 1000);
    assert.strictEqual(s.task.kvRestoredTokens, 800, 'KV REUSE = n_prompt_tokens_cache');
    assert.strictEqual(s.task.prefillTimingSeen, true, 'H7: 0/200 — измерено');

    // Фаза 2: task B (id 2) стартовал, а в полёте stale-ответ task A
    // (id_task=1 со счётчиками A) — он должен быть DISCARD
    mock.slotPayload = slotPayload({ idTask: 1, processing: true, prompt: 1000, processed: 200, cached: 800, decoded: 50 });
    child.stdin.write('slot operator(): id  0 | task 2 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 2000\n');
    s = await waitState(monPort, s => s.task.id === '2' && s.task.totalPrompt === 2000,
      'фаза 2: task B отслеживается из лога');
    await sleep(POLL_CYCLE); // ≥ один поллинг stale-ответа A ПОСЛЕ трекинга B
    s = await fetchState(monPort);
    assert.strictEqual(s.task.id, '2');
    assert.strictEqual(s.task.totalPrompt, 2000);
    assert.strictEqual(s.task.kvRestoredTokens, 0, 'stale A не перезаписал KV REUSE task B');
    assert.strictEqual(s.task.prefillTokens, 0, 'stale A не перезаписал PREFILL task B');
    assert.strictEqual(s.task.generatedTokens, 0, 'stale A не перезаписал DECODE task B');
    assert.strictEqual(s.n_prompt_tokens, 1000, 'occupancy не обновлён чужим ответом (DISCARD)');

    // Фаза 3: корректный ответ task B (id_task=2) — применяется
    mock.slotPayload = slotPayload({ idTask: 2, processing: true, prompt: 2000, processed: 300, cached: 1500, decoded: 70 });
    s = await waitState(monPort, s => s.task.kvRestoredTokens === 1500 && s.task.prefillTokens === 300 && s.task.generatedTokens === 70,
      'фаза 3: корректный ответ B применён');
    assert.strictEqual(s.task.id, '2');
    assert.strictEqual(s.n_prompt_tokens, 2000);
  } finally {
    child.kill('SIGKILL');
    srv.close();
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});

// П.1: значение KV, доказанное логом (kvEventEmitted), /slots перезаписать
// не способно — в обе стороны:
//   A) restored checkpoint (n_past=800) -> stale /slots cache=0 не сбрасывает;
//   B) full re-processing (kv=0) -> stale /slots cache=94676 (снимок
//      предыдущего task, слот получил task ДО prompt-processing) не прилипает.
test('п.1: kvEventEmitted-guard — /slots не перезаписывает KV, доказанный логом', async () => {
  const { mock, srv } = await startMockSlots();
  const { child, monPort, getStderr } = await startMonitor(mock.port);
  try {
    // Task A: лог доказывает KV (restored checkpoint, n_past = 800)
    mock.slotPayload = slotPayload({ idTask: 1, processing: true, prompt: 1000, processed: 200, cached: 800, decoded: 50 });
    child.stdin.write('slot operator(): id  0 | task 1 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000\n');
    child.stdin.write('slot operator(): id  0 | task 1 | restored context checkpoint (pos_min = 799, pos_max = 799, n_tokens = 800, n_past = 800, size = 120.000 MiB)\n');
    let s = await waitState(monPort, s => s.task.id === '1' && s.task.kvRestoredTokens === 800,
      'A: KV из лога (n_past)');
    assert.strictEqual(s.task.kvRestoredTokens, 800, 'KV из лога (n_past)');

    // Stale /slots с тем же id_task, но cache=0 (in-flight снимок до
    // prompt-processing) — доказанное значение не сбрасывается
    mock.slotPayload = slotPayload({ idTask: 1, processing: true, prompt: 1000, processed: 200, cached: 0, decoded: 50 });
    await sleep(POLL_CYCLE); // stale-ответ точно будет отдан на поллинг
    s = await fetchState(monPort);
    assert.strictEqual(s.task.kvRestoredTokens, 800, 'stale /slots (cache=0) не перезаписал лог');

    // Зеркало: task B — full re-processing (лог доказывает kv=0), а stale
    // /slots несёт cache=94676 предыдущего task — kv остаётся 0
    mock.slotPayload = slotPayload({ idTask: 2, processing: true, prompt: 2000, processed: 100, cached: 94676, decoded: 0 });
    child.stdin.write('slot operator(): id  0 | task 2 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 2000\n');
    child.stdin.write('slot operator(): id  0 | task 2 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)\n');
    s = await waitState(monPort, s => s.task.id === '2', 'B: отслеживается');
    await sleep(POLL_CYCLE); // stale 94676 точно будет отдан на поллинг
    s = await fetchState(monPort);
    assert.strictEqual(s.task.kvRestoredTokens, 0, 'full re-processing: kv=0, stale 94676 не прилип');
  } finally {
    child.kill('SIGKILL');
    srv.close();
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});

// П.2: prefillTokens неубывающий (Math.max) — stale in-flight снимок с
// НИЖЕ значением n_prompt_tokens_processed счётчик не откатывает.
test('п.2: prefillTokens — stale снимок ниже не откатывает (Math.max)', async () => {
  const { mock, srv } = await startMockSlots();
  const { child, monPort, getStderr } = await startMonitor(mock.port);
  try {
    mock.slotPayload = slotPayload({ idTask: 3, processing: true, prompt: 60000, processed: 50000, cached: 0, decoded: 0 });
    child.stdin.write('slot operator(): id  0 | task 3 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 60000\n');
    let s = await waitState(monPort, s => s.task.prefillTokens === 50000, '50000 применено');
    assert.strictEqual(s.task.prefillTokens, 50000);

    mock.slotPayload = slotPayload({ idTask: 3, processing: true, prompt: 60000, processed: 52000, cached: 0, decoded: 0 });
    s = await waitState(monPort, s => s.task.prefillTokens === 52000, '52000 применено');
    assert.strictEqual(s.task.prefillTokens, 52000);

    // Stale in-flight снимок с меньшим значением — отката нет
    mock.slotPayload = slotPayload({ idTask: 3, processing: true, prompt: 60000, processed: 51000, cached: 0, decoded: 0 });
    await sleep(POLL_CYCLE); // stale 51000 точно будет отдан на поллинг
    s = await fetchState(monPort);
    assert.strictEqual(s.task.prefillTokens, 52000, 'stale 51000 не откатил 52000');
    assert.strictEqual(s.task.prefillTimingSeen, true, 'H7: значение измерено');
  } finally {
    child.kill('SIGKILL');
    srv.close();
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});

// H7b: /slots processed=0 — НЕ доказательство измерения prefill
test('H7b: /slots processed=0 — prefillTimingSeen остаётся false', async () => {
  const { mock, srv } = await startMockSlots();
  const { child, monPort, getStderr } = await startMonitor(mock.port);
  try {
    mock.slotPayload = slotPayload({ idTask: 1, processing: true, prompt: 1000, processed: 0, cached: 1000, decoded: 0 });
    child.stdin.write('slot operator(): id  0 | task 1 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000\n');
    await waitState(monPort, s => s.task.id === '1', 'task отслеживается');
    await sleep(POLL_CYCLE); // ответ processed=0 точно будет отдан на поллинг
    const s = await fetchState(monPort);
    assert.strictEqual(s.task.prefillTokens, 0);
    assert.strictEqual(s.task.prefillTimingSeen, false, 'processed=0 — не доказательство измерения');
  } finally {
    child.kill('SIGKILL');
    srv.close();
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});
