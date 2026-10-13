'use strict';
// Устойчивость LIVE-монитора к некорректному входу (www.js):
//  1b — посторонние/битые строки в stdin не роняют и не портят движок
//       (try/catch вокруг handleLine + счётчик parseErrors в stderr);
//  2  — зависшее ТЕЛО ответа /slots (заголовки есть, тела нет) не должно
//       намертво фиксировать serverPollBusy: abort-таймер покрывает res.json().
// Запуск: node --test tests/
const { test } = require('node:test');
const assert = require('node:assert');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');

const WWW = path.join(__dirname, '..', 'www.js');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const POLL_CYCLE = 700; // ≥ один цикл pollServer(500 мс) + flushState(100 мс)

// Скриптуемый mock llama-server: mock.stallBody — заголовки приходят, тело нет.
function startMockSlots() {
  const mock = { stallBody: false, slotPayload: [] };
  const srv = http.createServer((req, res) => {
    if (req.url === '/slots') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      if (mock.stallBody) return; // тело не приходит никогда — клиент обязан отвалиться по таймеру
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

// 1b: мусор в stdin — монитор жив, движок после мусора работает, Uncaught нет.
test('1b: malformed stdin lines do not crash or corrupt the monitor', async () => {
  const { mock, srv } = await startMockSlots();
  const { child, monPort, getStderr } = await startMonitor(mock.port);
  try {
    const garbage = [
      'x'.repeat(100000), // аномально длинная строка
      '\ud800 одиночный суррогат',
      '\x00\x01\x02 управляющие символы',
      'slot operator(): id  0 | task 99 | new prompt, n_ctx_slot = abc, n_keep = 0, task.n_tokens = xyz',
      'tg = abc t/s, n_gen = 5',
      'srv update: - cache state: NaN prompts, Infinity MiB',
      'update: - prompt 0xZZ: notanumber tokens, checkpoints: x, y MiB',
      '[LCP-DEBUG] i = A = B = DIVERGE',
      "adding speculative implementation 'draft-mtp'",
      'n_max = not-a-number',
      'slot print_timing: id  0 | task 1 | prompt eval time =  ms / tokens',
      'eval time =   99864.69 ms /  tokens (   ms per token)'
    ];
    for (const g of garbage) child.stdin.write(g + '\n');
    await sleep(POLL_CYCLE); // мусор точно обработан (и записан в stdin-хендлере)

    // монитор отвечает
    await fetchState(monPort);
    // движок не испорчен: валидный таск применяется как обычно
    mock.slotPayload = slotPayload({ idTask: 1, processing: true, prompt: 1000, processed: 200, cached: 800, decoded: 50 });
    child.stdin.write('slot operator(): id  0 | task 1 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000\n');
    const s = await waitState(monPort, s => s.task.id === '1' && s.task.prefillTokens === 200 && s.task.kvRestoredTokens === 800,
      'движок intact после мусора');
    assert.strictEqual(s.serverOnline, true);
    assert.ok(!/Uncaught/.test(getStderr()), 'no uncaught exception: ' + getStderr());
  } finally {
    child.kill('SIGKILL');
    srv.close();
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});

// 2: stalled body — заголовки /slots приходят, тело зависает. Без abort-таймера
// на теле fetch не завершается никогда → serverPollBusy закреплён → все
// последующие поллинги пропускаются (монитор «замирает»). С фиксом: отвал по
// 400 мс, следующий поллинг проходит нормально.
test('2: stalled /slots body does not freeze the poll loop', async () => {
  const { mock, srv } = await startMockSlots();
  mock.stallBody = true; // с первого же поллинга: заголовки есть, тела нет
  const { child, monPort, getStderr } = await startMonitor(mock.port);
  try {
    // несколько циклов поллинга с зависшим телом (каждый обрывается по таймеру)
    await sleep(1500);
    // тело начинает отвечать — счётчики применятся только если busy НЕ закреплён
    mock.stallBody = false;
    mock.slotPayload = slotPayload({ idTask: 1, processing: true, prompt: 1000, processed: 200, cached: 800, decoded: 50 });
    child.stdin.write('slot operator(): id  0 | task 1 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000\n');
    const s = await waitState(monPort, s => s.n_prompt_tokens === 1000 && s.task.id === '1' && s.task.prefillTokens === 200,
      'poll loop recovered after stalled body');
    assert.strictEqual(s.serverOnline, true, 'успешный поллинг вернул ONLINE');
  } finally {
    child.kill('SIGKILL');
    srv.close();
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});
