'use strict';
// Тесты replay-режима www.js: детерминизм, UNKNOWN/EMPTY/USED,
// чекпоинты (create/evict/sweep по виртуальным часам), жизненный цикл таска.
// Запуск: node --test tests/
const { test } = require('node:test');
const assert = require('node:assert');
const { execFileSync, spawn } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const WWW = path.join(__dirname, '..', 'www.js');

// lines: [{ ts: '2026-09-04T09:46:12.656Z', msg: '...' }]
function replay(lines) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-test-'));
  const log = path.join(dir, 'log.txt');
  const out = path.join(dir, 'out.json');
  fs.writeFileSync(log, lines.map(l => `[${l.ts}] ${l.msg}`).join('\n') + '\n');
  execFileSync('node', [WWW, '--replay', log, out], {
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' },
    stdio: 'ignore'
  });
  return JSON.parse(fs.readFileSync(out, 'utf8'));
}

const T = (s) => `2026-09-04T09:46:${String(s).padStart(2, '0')}.000Z`; // секунды от 09:46:00

test('determinism: two replays of the same log are identical', () => {
  const lines = [
    { ts: T(0), msg: 'srv update: - cache state: 3 prompts, 12.500 MiB (limits: 40960.000 MiB, 123136 tokens, 42949672960 est)' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' },
    { ts: T(5), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ];
  assert.deepStrictEqual(replay(lines), replay(lines));
});

test('UNKNOWN: no "cache state:" line -> null ramCache fields', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.ramCache.limitMiB, null);
  assert.strictEqual(s.ramCache.usedMiB, null);
  assert.strictEqual(s.ramCache.promptsCount, null);
  assert.strictEqual(s.ramCache.promptCacheMiB, null);
  assert.strictEqual(s.ramCache.cpMax, null);
});

test('EMPTY: "cache state: 0 prompts, 0 MiB" -> zeros, not null', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 0 prompts, 0.000 MiB (limits: 40960.000 MiB, 123136 tokens, 42949672960 est)' }
  ]);
  assert.strictEqual(s.ramCache.promptsCount, 0);
  assert.strictEqual(s.ramCache.usedMiB, 0);
  assert.strictEqual(s.ramCache.limitMiB, 40960);
  assert.strictEqual(s.ramCache.promptCacheMiB, 0);
});

test('checkpoint create: slots, cpMax, cpCurrent, cpUsedMiB', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' }
  ]);
  assert.strictEqual(s.ramCache.cpMax, 32);
  assert.strictEqual(s.ramCache.cpCreated, 1);
  assert.strictEqual(s.ramCache.cpCurrent, 1);
  assert.strictEqual(s.ramCache.cpUsedMiB, 1.5);
  // createdAt/restoreCount детерминированы виртуальными часами и остаются в снапшоте
  assert.deepStrictEqual(s.ramCache.slots['0:100:100'], { key: '0:100:100', slot: 0, tokens: 101, posMin: 100, posMax: 100, size: 1.5, createdAt: Date.parse(T(0)), lastRestored: null, restoreCount: 0 });
});

test('evict + sweep: slot removed at next checkpoint line >2s later (log time)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' },
    { ts: T(4), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  assert.strictEqual(s.ramCache.cpErased, 1);
  assert.deepStrictEqual(Object.keys(s.ramCache.slots), ['0:200:200']);
  assert.strictEqual(s.ramCache.cpCurrent, 2); // N=2 авторитетен (resync), живой слот один
  assert.strictEqual(s.ramCache.cpUsedMiB, 2.0);
});

test('sweep boundary: evicted slot kept when next checkpoint line <2s later', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  // 1000 мс < 2000 мс → слот ещё на стойке (evictedAt вырезан из снапшота)
  assert.deepStrictEqual(Object.keys(s.ramCache.slots).sort(), ['0:100:100', '0:200:200']);
  assert.strictEqual(s.ramCache.cpCurrent, 2); // N=2 авторитетен (resync), живой слот один
  assert.strictEqual(s.ramCache.cpUsedMiB, 2.0);
});

test('evict via "erasing old": capacity eviction of oldest checkpoint (D7)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | erasing old context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(4), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  assert.strictEqual(s.ramCache.cpErased, 1);
  assert.deepStrictEqual(Object.keys(s.ramCache.slots), ['0:200:200']);
  assert.strictEqual(s.ramCache.cpCurrent, 2); // N=2 авторитетен (resync), живой слот один
  assert.strictEqual(s.ramCache.cpUsedMiB, 2.0);
  const ev = s.events.find(e => e.tag === 'KV EVICT RAM');
  assert.ok(ev, 'KV EVICT RAM event present');
  assert.strictEqual(ev.params.reasonKey, 'reason_capacity', 'reason — capacity (oldest evicted on fill)');
  assert.strictEqual(ev.params.tok, '101', 'токены из строки');
});

test('evict via "erasing old" without n_tokens: fallback pos_max + 1 (D7)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | erasing old context checkpoint (pos_min = 100, pos_max = 100, size = 1.500 MiB)' },
    { ts: T(4), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  assert.strictEqual(s.ramCache.cpErased, 1);
  assert.deepStrictEqual(Object.keys(s.ramCache.slots), ['0:200:200']);
  const ev = s.events.find(e => e.tag === 'KV EVICT RAM');
  assert.strictEqual(ev.params.tok, '101', 'pos_max(100) + 1');
});

// H8: точный identity чекпоинта = (slot, pos_min, pos_max), а НЕ n_tokens.
// Два чекпоинта одного слота с ОДИНАКОВЫМ n_tokens, но разными позициями
// (SWA-окно) — это РАЗНЫЕ объекты upstream; старый ключ slot:n_tokens
// перезаписывал A объектом B. Erase по pos_min/pos_max снимает ТОЛЬКО свой.
test('H8: одинаковые n_tokens, разные pos_min/pos_max — два разных чекпоинта', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    // тот же n_tokens=101, но другие позиции — отдельный объект
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 150, pos_max = 150, n_tokens = 101, size = 1.600 MiB)' },
    // erase первого (pos 100/100) — второй (pos 150/150) должен выжить
    { ts: T(2), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' }
  ]);
  assert.deepStrictEqual(Object.keys(s.ramCache.slots).sort(), ['0:100:100', '0:150:150']);
  assert.ok(s.ramCache.slots['0:100:100'].removedAt, 'первый снят');
  assert.strictEqual(s.ramCache.slots['0:150:150'].removedAt, undefined, 'второй жив');
  assert.strictEqual(s.ramCache.slots['0:150:150'].tokens, 101);
});

test('volatile keys stripped from snapshot (serverNow, ts, event ids)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.serverNow, undefined);
  for (const e of s.events) {
    assert.strictEqual(e.id, undefined);
    assert.strictEqual(e.ts, undefined);
  }
});

test('task lifecycle: new prompt -> prefill -> decode -> eval time -> history', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 96015' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 327 | prompt eval time =  178222.54 ms / 96015 tokens (    1.86 ms per token,   538.74 tokens per second)' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 327 | n_gen =    101, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 327 |        eval time =   99864.69 ms /  2574 tokens (   38.81 ms per token,    25.76 tokens per second)' }
  ]);
  assert.strictEqual(s.history.length, 1);
  const h = s.history[0];
  assert.strictEqual(h.id, '327');
  assert.strictEqual(h.input, 96015);
  assert.strictEqual(h.prefill, 96015);
  assert.strictEqual(h.prefillSource, 'measured');
  assert.strictEqual(h.decode, 2574);
  assert.strictEqual(h.speed, 25.76);
  assert.strictEqual(h.total, 96015 + 2574);
  assert.strictEqual(h.isTruncated, false);
});

test('two consecutive tasks: task A cannot contaminate task B (replay)', () => {
  const s = replay([
    // Task A: prompt 1000, KV reuse 800, prefill 200, decode 50
    { ts: T(0), msg: 'slot operator(): id  0 | task 101 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 101 | cached n_tokens = 800, memory_seq_rm [800, end)' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 101 | prompt eval time =  100.00 ms / 200 tokens (    0.50 ms per token,  2000.00 tokens per second)' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 101 |        eval time =   500.00 ms /   50 tokens (   10.00 ms per token,    50.00 tokens per second)' },
    // Task B: prompt 2000, cold (no cache), prefill 2000, decode 70
    { ts: T(5), msg: 'slot operator(): id  0 | task 102 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 2000' },
    { ts: T(6), msg: 'slot print_timing: id  0 | task 102 | prompt eval time =  200.00 ms / 2000 tokens (    0.10 ms per token,  1000.00 tokens per second)' },
    { ts: T(7), msg: 'slot print_timing: id  0 | task 102 |        eval time =   700.00 ms /   70 tokens (   10.00 ms per token,    70.00 tokens per second)' }
  ]);
  assert.strictEqual(s.history.length, 2);
  const b = s.history[0], a = s.history[1];
  // Task B — чистая строка, без следов task A
  assert.strictEqual(b.id, '102');
  assert.strictEqual(b.input, 2000);
  assert.strictEqual(b.kvReuse, 0);
  assert.strictEqual(b.prefill, 2000);
  assert.strictEqual(b.prefillSource, 'measured');
  assert.strictEqual(b.decode, 70);
  assert.strictEqual(b.speed, 70);
  assert.strictEqual(b.total, 2070);
  // Task A — свои значения, не затёрты task B
  assert.strictEqual(a.id, '101');
  assert.strictEqual(a.input, 1000);
  assert.strictEqual(a.kvReuse, 800);
  assert.strictEqual(a.prefill, 200);
  assert.strictEqual(a.prefillSource, 'measured');
  assert.strictEqual(a.decode, 50);
  assert.strictEqual(a.total, 1050);
});

test('entry then gone: prompt entry appears in cache state, then disappears (R1)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 1 prompts, 100.000 MiB (limits: 40960.000 MiB, 123136 tokens, 100000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' },
    { ts: T(1), msg: 'srv update: - cache state: 0 prompts, 0.000 MiB (limits: 40960.000 MiB, 123136 tokens, 0 est)' },
    { ts: T(1), msg: 'srv get_availabl: prompt cache update took 5.00 ms' }
  ]);
  assert.strictEqual(s.ramCache.limitTokens, 123136);
  assert.strictEqual(s.ramCache.entries.length, 1);
  const e = s.ramCache.entries[0];
  assert.strictEqual(e.key, '0xabc');
  assert.strictEqual(e.tokens, 1000);
  assert.strictEqual(e.checkpoints, 2);
  assert.strictEqual(e.sizeMiB, 100);
  assert.ok(e.removedAt, 'entry marked removed');
  // Первый коммит — baseline: createdAt=null (B6), вспышки при старте нет
  assert.strictEqual(e.createdAt, null);
});

test('P5: эффективный токен-лимит = max(configured, est) из строки cache state', () => {
  // est > configured: сервер динамически поднял лимит (limit_size/size_per_token)
  const s1 = replay([
    { ts: T(0), msg: 'srv update: - cache state: 1 prompts, 100.000 MiB (limits: 40960.000 MiB, 123136 tokens, 200000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  assert.strictEqual(s1.ramCache.limitTokens, 123136, 'configured сохранён');
  assert.strictEqual(s1.ramCache.limitTokensEff, 200000, 'эффективный = est (est > configured)');

  // est < configured (несогласованный снапшот) — max берёт configured
  const s2 = replay([
    { ts: T(0), msg: 'srv update: - cache state: 0 prompts, 0.000 MiB (limits: 40960.000 MiB, 123136 tokens, 0 est)' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 5.00 ms' }
  ]);
  assert.strictEqual(s2.ramCache.limitTokensEff, 123136, 'эффективный = configured (est < configured)');

  // Старый формат без «est» — эффективный = configured
  const s3 = replay([
    { ts: T(0), msg: 'srv update: - cache state: 0 prompts, 0.000 MiB (limits: 40960.000 MiB, 123136 tokens)' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 5.00 ms' }
  ]);
  assert.strictEqual(s3.ramCache.limitTokensEff, 123136, 'без est — configured');
});

test('commit: identical re-commit does not set touchedAt, changed sizeMiB does (B1)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 1 prompts, 100.000 MiB (limits: 40960.000 MiB, 123136 tokens, 100000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' },
    { ts: T(1), msg: 'srv update: - cache state: 1 prompts, 100.000 MiB (limits: 40960.000 MiB, 123136 tokens, 100000 est)' },
    { ts: T(1), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(1), msg: 'srv get_availabl: prompt cache update took 10.00 ms' },
    { ts: T(2), msg: 'srv update: - cache state: 1 prompts, 150.000 MiB (limits: 40960.000 MiB, 123136 tokens, 150000 est)' },
    { ts: T(2), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 150.000 MiB' },
    { ts: T(2), msg: 'srv get_availabl: prompt cache update took 10.00 ms' },
    { ts: T(3), msg: 'srv update: - cache state: 1 prompts, 150.000 MiB (limits: 40960.000 MiB, 123136 tokens, 150000 est)' },
    { ts: T(3), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 150.000 MiB' },
    { ts: T(3), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  const e = s.ramCache.entries[0];
  assert.strictEqual(e.sizeMiB, 150);
  // touchedAt выставлен только при реальном изменении (T(2));
  // идентичные коммиты T(1) и T(3) его не тронули
  assert.strictEqual(e.touchedAt, Date.parse(T(2)));
});

test('first commit: baseline -> createdAt null, no born effect (B6)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 1 prompts, 100.000 MiB (limits: 40960.000 MiB, 123136 tokens, 100000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  assert.strictEqual(s.ramCache.entries[0].createdAt, null);
  assert.strictEqual(s.ramCache.entries[0].touchedAt, null);
  const model = ringModel(s);
  assert.strictEqual(model.until, 0); // вспышек при старте нет
});

test('entry without 0x prefix and uppercase address parses (B7)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 1 prompts, 172.505 MiB (limits: 40960.000 MiB, 123136 tokens, 100000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 000001E6B5CEA6B0: 11869 tokens, checkpoints: 0, 172.505 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  assert.strictEqual(s.ramCache.entries.length, 1);
  assert.strictEqual(s.ramCache.entries[0].key, '000001E6B5CEA6B0');
  assert.strictEqual(s.ramCache.entries[0].tokens, 11869);
  assert.strictEqual(s.ramCache.entries[0].checkpoints, 0);
  assert.strictEqual(s.ramCache.entries[0].sizeMiB, 172.505);
});

test('N==1 clears: new slot with N=1 resyncs away older live slots (R1)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 328 | created context checkpoint 1 of 32 (pos_min = 300, pos_max = 300, n_tokens = 301, size = 2.500 MiB)' }
  ]);
  assert.deepStrictEqual(Object.keys(s.ramCache.slots).sort(), ['0:100:100', '0:200:200', '0:300:300']);
  assert.strictEqual(s.ramCache.slots['0:300:300'].removedAt, undefined);
  assert.strictEqual(s.ramCache.slots['0:100:100'].removedAt, s.ramCache.slots['0:200:200'].removedAt);
  assert.strictEqual(s.ramCache.slots['0:100:100'].removeReason, 'resync');
  assert.strictEqual(s.ramCache.cpCurrent, 1);
  assert.strictEqual(s.ramCache.cpUsedMiB, 2.5);
});

// --ring-model: снимок состояния -> модель колец (чистая функция buildRingModel)
function ringModel(s) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-rm-'));
  const f = path.join(dir, 'state.json');
  fs.writeFileSync(f, JSON.stringify(s));
  const out = execFileSync('node', [WWW, '--ring-model', f], {
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' },
    stdio: ['ignore', 'pipe', 'ignore']
  }).toString();
  return JSON.parse(out);
}

test('ring model: ring 3 items/k, live total == promptCacheMiB / limitMiB (R3 gate)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 2 prompts, 300.000 MiB (limits: 40960.000 MiB, 123136 tokens, 300000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xaaa:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'srv update:   - prompt 0xbbb:  2000 tokens, checkpoints: 1, 200.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  const model = ringModel(s);
  // Кольцо 3: упорядоченный список items + масштаб k; сумма длин живых
  // дуг (sizeMiB * k / limit) точно равна promptCacheMiB / limitMiB
  assert.strictEqual(model.ram.items.length, 2);
  assert.strictEqual(model.ram.items[0].key, '0xaaa');
  assert.strictEqual(model.ram.items[1].key, '0xbbb');
  assert.strictEqual(model.ram.k, 1); // used(300) == sumLive(300)
  const total = model.ram.items.reduce((a, it) => a + it.sizeMiB * model.ram.k / model.ram.limit, 0);
  const expected = s.ramCache.promptCacheMiB / s.ramCache.limitMiB;
  assert.ok(Math.abs(total - expected) < 1e-9, `total=${total} expected=${expected}`);
  // Кольцо 2: нет created-строк -> cpMax null -> state 'na'
  assert.strictEqual(model.cp.state, 'na');
  assert.strictEqual(model.cp.marks.length, 0);
});

test('ring model: ring 2 marks at tokens/n_ctx, live slots only (R3)', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' },
    { ts: T(6), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 3 of 32 (pos_min = 300, pos_max = 300, n_tokens = 301, size = 2.500 MiB)' }
  ]);
  const model = ringModel(s);
  assert.strictEqual(model.cp.state, 'ok');
  assert.strictEqual(model.cp.max, 32);
  // Живые слоты: 101 вытеснен (removedAt), 201 и 301 живы
  const toks = model.cp.marks.map(m => m.tokens);
  assert.deepStrictEqual(toks, [201, 301]);
  // frac = tokens / n_ctx
  assert.ok(Math.abs(model.cp.marks[0].frac - 201 / s.n_ctx) < 1e-12);
  assert.ok(Math.abs(model.cp.marks[1].frac - 301 / s.n_ctx) < 1e-12);
});

test('ring model: confirmed zero -> solid empty track (state ok, no items) (R3)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 0 prompts, 0.000 MiB (limits: 40960.000 MiB, 123136 tokens, 0 est)' }
  ]);
  const model = ringModel(s);
  // limitMiB задан (0 prompts — подтверждённый ноль, не null) -> state 'ok',
  // сплошной пустой трек, без N/A
  assert.strictEqual(model.ram.state, 'ok');
  assert.strictEqual(model.ram.limit, 40960);
  assert.strictEqual(model.ram.used, 0);
  assert.strictEqual(model.ram.items.length, 0);
});

test('ring model: dying checkpoint in 2s window carries removedAt (R4)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' }
  ]);
  const model = ringModel(s);
  // 101 вытеснен в T(2), снимок сделан в T(2) — внутри 2-секундного окна
  const toks = model.cp.marks.map(m => m.tokens);
  assert.deepStrictEqual(toks, [101, 201]);
  assert.strictEqual(model.cp.marks[0].removedAt, s.ramCache.slots['0:100:100'].evictedAt);
  assert.strictEqual(model.cp.marks[1].removedAt, null);
  // until включает death умирающего (2000 мс)
  assert.ok(model.until >= s.ramCache.slots['0:100:100'].removedAt + 2000);
});

test('ring model: until = end of last effect (R4)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(10), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  const model = ringModel(s);
  // Последний эффект: born чекпоинта 201 (FX.born = 900 мс)
  assert.strictEqual(model.until, s.ramCache.slots['0:200:200'].createdAt + 900);
});

test('ring model: cache entry until = createdAt + grow + born (R4)', () => {
  // Первый коммит пустой (baseline), запись приходит вторым коммитом —
  // createdAt = время второго коммита, born-эффект активен
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 0 prompts, 0.000 MiB (limits: 40960.000 MiB, 123136 tokens, 0 est)' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' },
    { ts: T(10), msg: 'srv update: - cache state: 1 prompts, 100.000 MiB (limits: 40960.000 MiB, 123136 tokens, 100000 est)' },
    { ts: T(10), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(10), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  const model = ringModel(s);
  assert.strictEqual(s.ramCache.entries[0].createdAt, Date.parse(T(10)));
  // born записи кэша: дуга вырастает (grow=500), затем вспышка (born=900)
  assert.strictEqual(model.until, s.ramCache.entries[0].createdAt + 500 + 900);
});

test('ring model: removed entry -> items with removedAt, live sum == used/limit (R4)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 2 prompts, 300.000 MiB (limits: 40960.000 MiB, 123136 tokens, 300000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xaaa:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'srv update:   - prompt 0xbbb:  2000 tokens, checkpoints: 1, 200.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' },
    { ts: T(1), msg: 'srv update: - cache state: 1 prompts, 200.000 MiB (limits: 40960.000 MiB, 123136 tokens, 200000 est)' },
    { ts: T(1), msg: 'srv update:   - prompt 0xbbb:  2000 tokens, checkpoints: 1, 200.000 MiB' },
    { ts: T(1), msg: 'srv get_availabl: prompt cache update took 5.00 ms' }
  ]);
  const model = ringModel(s);
  // 0xaaa удалена в T(1) (1 с назад, внутри окна 2с) — остаётся в items с removedAt
  const dying = model.ram.items.filter(i => i.removedAt);
  assert.strictEqual(dying.length, 1);
  assert.strictEqual(dying[0].key, '0xaaa');
  assert.strictEqual(dying[0].removedAt, s.ramCache.entries.find(e => e.key === '0xaaa').removedAt);
  // Живая запись: 0xbbb; сумма длин живых == used/limit (гейт R3 сохранён)
  const live = model.ram.items.filter(i => !i.removedAt);
  assert.strictEqual(live.length, 1);
  assert.strictEqual(live[0].key, '0xbbb');
  const total = live.reduce((a, it) => a + it.sizeMiB * model.ram.k / model.ram.limit, 0);
  assert.ok(Math.abs(total - s.ramCache.promptCacheMiB / s.ramCache.limitMiB) < 1e-9);
  // until включает death умирающей записи (2000 мс)
  assert.ok(model.until >= dying[0].removedAt + 2000);
});

// --play: time-lapse воспроизведение лога (сервер поднимается на случайном
// порту и остаётся работать после конца лога — резолвим по маркеру, потом
// SIGKILL). Случайный порт: тесты не сталкиваются с живым монитором/CI на 3000.
function playLog(lines, { timeoutMs = 20000 } = {}) {
  const { spawn } = require('node:child_process');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-play-'));
  const log = path.join(dir, 'log.txt');
  fs.writeFileSync(log, lines.map(l => (typeof l === 'string' ? l : `[${l.ts}] ${l.msg}`)).join('\n') + '\n');
  const monPort = 40000 + Math.floor(Math.random() * 9000);
  const child = spawn('node', [WWW, '--play', log, '--speed', '1000', '--port', String(monPort)], {
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let out = '';
  let done = false;
  return new Promise((resolve) => {
    const finish = (timedOut) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      const code = child.exitCode;
      child.kill('SIGKILL');
      resolve({ code, out, timedOut });
    };
    const timer = setTimeout(() => finish(true), timeoutMs);
    child.stdout.on('data', (d) => {
      out += d;
      if (out.includes('Playback finished')) finish(false);
    });
    child.stderr.on('data', (d) => { out += d; });
    child.on('close', (code) => { child.exitCode = code; finish(false); });
  });
}

test('B5: --play skips untimestamped lines without hanging', async () => {
  const lines = [
    '', // пустая первая строка
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' },
    'no timestamp here', // неразмеченная строка в середине
    { ts: T(5), msg: 'slot print_timing: id  0 | task 327 | n_gen =    101, tg =  22.70 t/s, tg_3s = 22.92 t/s' }
  ];
  const r = await playLog(lines);
  assert.ok(!r.timedOut, 'play hung: ' + r.out.slice(-300));
  assert.ok(r.out.includes('Playback finished'), r.out.slice(-500));
});

test('B5: --play without any timestamps exits 1 with LLAMA_LOG_FILE error', async () => {
  const r = await playLog(['hello', 'world']);
  assert.strictEqual(r.code, 1);
  assert.ok(r.out.includes('LLAMA_LOG_FILE'), r.out);
});

test('B9: --replay takes n_ctx from n_ctx_slot in new prompt line', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.n_ctx, 123136);
});

test('B10a: resync keeps the just-created slot on createdAt tie', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 49999, pos_max = 49999, n_tokens = 50000, size = 50.000 MiB)' },
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 59999, pos_max = 59999, n_tokens = 60000, size = 60.000 MiB)' },
    { ts: T(0), msg: 'slot create_check: id  0 | task 328 | created context checkpoint 1 of 32 (pos_min = 4999, pos_max = 4999, n_tokens = 5000, size = 0.500 MiB)' }
  ]);
  // Все три createdAt равны T(0) — без keepKey сортировка могла снять
  // только что созданный слот 5000
  assert.strictEqual(s.ramCache.slots['0:4999:4999'].removedAt, undefined);
  assert.strictEqual(s.ramCache.slots['0:49999:49999'].removeReason, 'resync');
  assert.strictEqual(s.ramCache.slots['0:59999:59999'].removeReason, 'resync');
  assert.strictEqual(s.ramCache.cpCurrent, 1);
  assert.strictEqual(s.ramCache.cpUsedMiB, 0.5);
});

test('B10b: restore fuzzy search skips dead slots', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 700, pos_max = 700, n_tokens = 701, size = 7.000 MiB)' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 327 | restored context checkpoint (pos_min = 104, pos_max = 104, n_tokens = 105, n_past = 105, size = 1.500 MiB)' }
  ]);
  // 101 мёртв (evicted в T(1)) — fuzzy |101-105|=4 < 500 не должен его
  // подсветить; 701 слишком далеко (|701-105|=596 > 500)
  assert.strictEqual(s.ramCache.slots['0:100:100'].restoreCount, 0);
  assert.strictEqual(s.ramCache.slots['0:700:700'].restoreCount, 0);
});

test('B2 gate: computeRing3Layout — dying arc shrinks in place, live arcs slide, no overlaps', () => {
  // Чистая функция раскладки извлекается из исходника www.js и проверяется
  // напрямую: от removedAt до removedAt+2000 с шагом 100 мс дуги не
  // пересекаются, сумма длин живых дуг == used/limit.
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/function computeRing3Layout[\s\S]*?\n}/);
  assert.ok(m, 'computeRing3Layout found in www.js');
  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
  const FX = { born: 900, touch: 8000, death: 2000, beam: 1500, grow: 500 };
  const computeRing3Layout = new Function('FX', 'clamp01', m[0] + '\nreturn computeRing3Layout;')(FX, clamp01);

  const limit = 40960;
  const removedAt = Date.parse(T(1));
  const ram = {
    limit: limit,
    used: 200,
    k: 1, // used(200) == sumLive(200)
    items: [
      { key: '0xaaa', tokens: 1000, checkpoints: 2, sizeMiB: 100, createdAt: null, touchedAt: null, removedAt: removedAt },
      { key: '0xbbb', tokens: 2000, checkpoints: 1, sizeMiB: 200, createdAt: null, touchedAt: null, removedAt: null }
    ]
  };
  for (let now = removedAt; now <= removedAt + 2000; now += 100) {
    const { arcs } = computeRing3Layout(ram, now, true);
    // Дуги не пересекаются: start[i] >= end[i-1]
    let prevEnd = 0;
    for (const a of arcs) {
      assert.ok(a.start >= prevEnd - 1e-12, `overlap at now=${now}: start=${a.start} < prevEnd=${prevEnd}`);
      prevEnd = a.start + a.len;
    }
    // Сумма длин живых дуг == used/limit
    const liveSum = arcs.filter(a => !a.removedAt).reduce((s, a) => s + a.len, 0);
    assert.ok(Math.abs(liveSum - ram.used / limit) < 1e-9, `liveSum=${liveSum} at now=${now}`);
  }
  // После окна смерти умирающая дуга исчезает
  const after = computeRing3Layout(ram, removedAt + 2001, true);
  assert.strictEqual(after.arcs.length, 1);
  assert.strictEqual(after.arcs[0].key, '0xbbb');
});

// ===== N-раунд (ревью версии B1–B10) =====

test('N1 gate: computeRing3Layout — кэш ~95%, умирающая дуга: сумма len <= 1 на всех кадрах', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/function computeRing3Layout[\s\S]*?\n}/);
  assert.ok(m, 'computeRing3Layout found in www.js');
  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
  const FX = { born: 900, touch: 8000, death: 2000, beam: 1500, grow: 500 };
  const computeRing3Layout = new Function('FX', 'clamp01', m[0] + '\nreturn computeRing3Layout;')(FX, clamp01);

  const limit = 40960;
  const used = 38912; // ~95% лимита
  const removedAt = Date.parse(T(1));
  const ram = {
    limit: limit,
    used: used,
    k: used / 36912, // живая запись 36912 MiB
    items: [
      { key: '0xaaa', tokens: 1000, checkpoints: 2, sizeMiB: 2000, createdAt: null, touchedAt: null, removedAt: removedAt },
      { key: '0xbbb', tokens: 2000, checkpoints: 1, sizeMiB: 36912, createdAt: null, touchedAt: null, removedAt: null }
    ]
  };
  for (let now = removedAt; now <= removedAt + 2000; now += 100) {
    const { arcs } = computeRing3Layout(ram, now, true);
    const sum = arcs.reduce((s, a) => s + a.len, 0);
    assert.ok(sum <= 1 + 1e-9, `sum=${sum} > 1 at now=${now}`);
    let prevEnd = 0;
    for (const a of arcs) {
      assert.ok(a.start >= prevEnd - 1e-12, `overlap at now=${now}`);
      prevEnd = a.start + a.len;
    }
  }
});

test('N2 gate: посторонняя строка между записями не коммитит буфер (нет ложных removedAt)', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 2 prompts, 300.000 MiB (limits: 40960.000 MiB, 123136 tokens, 300000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xaaa:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'some unrelated server line' },
    { ts: T(0), msg: 'srv update:   - prompt 0xbbb:  2000 tokens, checkpoints: 1, 200.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  assert.strictEqual(s.ramCache.entries.length, 2);
  for (const e of s.ramCache.entries) {
    assert.strictEqual(e.removedAt, null, e.key + ' got false removedAt');
  }
});

test('N3 gate: limits: -1.000 MiB -> unlimited, кольцо масштабируется по used', () => {
  const s = replay([
    { ts: T(0), msg: 'srv update: - cache state: 1 prompts, 100.000 MiB (limits: -1.000 MiB, 123136 tokens, 100000 est)' },
    { ts: T(0), msg: 'srv update:   - prompt 0xabc:  1000 tokens, checkpoints: 2, 100.000 MiB' },
    { ts: T(0), msg: 'srv get_availabl: prompt cache update took 10.00 ms' }
  ]);
  assert.strictEqual(s.ramCache.limitMiB, -1);
  const model = ringModel(s);
  assert.strictEqual(model.ram.state, 'unlimited');
  assert.strictEqual(model.ram.limit, 100); // limitEff = used
  assert.strictEqual(model.ram.items.length, 1);
});

test('N4 gate: --play виртуальные часы идут в реальном времени (не прыгают по таймстемпам)', async () => {
  const { spawn } = require('node:child_process');
  const http = require('node:http');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-play-'));
  const log = path.join(dir, 'log.txt');
  // Чекпоинт в T(0), второй снимок через 100 с лог-времени: при --speed 10
  // вторая строка придёт через ~10 с реального времени. Старые часы прыгали
  // на +100 с. (T(100) невалиден — секунды > 59; пишем ISO явно.)
  const T100 = '2026-09-04T09:47:40.000Z';
  fs.writeFileSync(log,
    `[${T(0)}] slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)\n` +
    `[${T100}] srv update: - cache state: 1 prompts, 100.000 MiB (limits: 40960.000 MiB, 123136 tokens, 100000 est)\n`);
  const port = 30000 + Math.floor(Math.random() * 20000); // случайный порт: не сталкиваться с живым монитором на 3000
  const child = spawn('node', [WWW, '--play', log, '--speed', '10', '--port', String(port)], {
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let out = '';
  try {
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('play timeout: ' + out.slice(-300))), 30000);
      child.stdout.on('data', (d) => {
        out += d;
        if (out.includes('Playback finished')) { clearTimeout(timer); resolve(); }
      });
      child.stderr.on('data', (d) => { out += d; });
    });
    const state = await new Promise((resolve, reject) => {
      http.get(`http://127.0.0.1:${port}/api/state`, (res) => {
        let b = '';
        res.on('data', (d) => { b += d; });
        res.on('end', () => resolve(JSON.parse(b)));
      }).on('error', reject);
    });
    const created = state.ramCache.slots['0:100:100'].createdAt;
    const delta = state.serverNow - created;
    // ~10 с реального времени (100 с лог-времени / speed 10), а не 100 с
    assert.ok(delta > 8000 && delta < 20000, `serverNow - createdAt = ${delta} ms (ожидалось ~10000)`);
  } finally {
    child.kill('SIGKILL');
  }
});

test('N5: restore пульсирует точный живой слот (n_tokens совпадает)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | restored context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_past = 101, size = 1.500 MiB)' }
  ]);
  assert.strictEqual(s.ramCache.slots['0:100:100'].restoreCount, 1);
  assert.strictEqual(s.ramCache.slots['0:100:100'].lastRestored, Date.parse(T(1)));
});

test('N5: restore точного мёртвого слота не пульсирует', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 327 | restored context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_past = 101, size = 1.500 MiB)' }
  ]);
  assert.strictEqual(s.ramCache.slots['0:100:100'].restoreCount, 0);
});

test('N6: вытеснение ставит removeReason (invalidated / capacity)', () => {
  const s1 = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  assert.strictEqual(s1.ramCache.slots['0:100:100'].removeReason, 'invalidated');
  const s2 = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | erasing old context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  assert.strictEqual(s2.ramCache.slots['0:100:100'].removeReason, 'capacity');
});

test('N7: метка кольца 2 не тоньше 3px (half = max(0.006, 3/R2)), hover-порог 0.03', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/function paintRing2[\s\S]*?\n    }\n/);
  assert.ok(m, 'paintRing2 found in www.js');
  assert.ok(m[0].includes('const half = Math.max(0.006, 3 / R2)'), 'half вычисляется');
  assert.ok(!m[0].includes('a - 0.006'), 'нет хардкода 0.006');
  assert.ok(src.includes('let best = null, bestD = 0.03;'), 'hover-порог 0.03');
});

// V1: n_ctx — только подтверждённые значения (строка сервера / /slots), без fallback 100096
test('V1: n_ctx = null, пока сервер не напечатал n_ctx_slot', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' }
  ]);
  assert.strictEqual(s.n_ctx, null);
});

test('V1: n_ctx из стартовой строки сервера (initializing, n_ctx_slot = 123136)', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" }
  ]);
  assert.strictEqual(s.n_ctx, 123136);
});

test('V1: n_ctx из строки new prompt (n_ctx_slot = 123136)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.n_ctx, 123136);
});

test('V1: в www.js нет fallback 100096', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(!src.includes('100096'), 'fallback 100096 удалён');
});

// V2: slotSource и checkpointRestore ортогональны — чекпоинт НЕ переопределяет источник
test('V2: hot-vram + checkpoint (паттерн 131): cached -> restored, оба флага', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.999 (90609/90714) > 0.100' },
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.999 (> 0.100 thold), f_keep = 1.000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 4659 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 90714' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 4659 | cached n_tokens = 90424, memory_seq_rm [90424, end)' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 4659 | restored context checkpoint (pos_min = 90423, pos_max = 90423, n_tokens = 90424, n_past = 90424, size = 504.915 MiB)' }
  ]);
  assert.strictEqual(s.task.slotSource, 'hot-vram');
  assert.strictEqual(s.task.checkpointRestore, true);
  assert.strictEqual(s.task.kvRestoredTokens, 90424);
});

test('V2: prompt-cache-ram + checkpoint (таск 5769): found better prompt + restored, оба флага', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.002 (32/16417) > 0.100' },
    { ts: T(1), msg: 'srv          load:    - prompt with length    7502, lcp =    7463, f_keep = 0.995, f_sim = 0.455' },
    { ts: T(1), msg: 'srv          load:  - found better prompt with f_keep = 0.995, f_sim = 0.455' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 5769 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 16417' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 5769 | restored context checkpoint (pos_min = 7348, pos_max = 7348, n_tokens = 7349, n_past = 7349, size = 178.501 MiB)' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 5769 | cached n_tokens = 7349, memory_seq_rm [7349, end)' }
  ]);
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram'); // чекпоинт НЕ затирает источник
  assert.strictEqual(s.task.checkpointRestore, true);
  assert.strictEqual(s.task.kvRestoredTokens, 7349);
  assert.strictEqual(s.task.lcpTokens, 7463);
});

// P1: LCP-кандидаты ключируются по slot_id — интерливинг двух слотов не
// перетекает LCP одного таска в другой (глобальная pendingLcp это делала).
test('P1: interleave двух слотов — LCP не перетекает между тасками (slot_id keying)', () => {
  const s = replay([
    // slot 0: LCP 90000/100000
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.900 (90000/100000) > 0.100' },
    // slot 1: LCP 50000/100000 — приходит ДО new prompt слота 0 (интерливинг)
    { ts: T(1), msg: 'slot get_availabl: id  1 | task -1 |  - checking sim = 0.500 (50000/100000) > 0.100' },
    // slot 0: task 100 → должен получить СВОЙ LCP 90000, не 50000 от слота 1
    { ts: T(2), msg: 'slot operator(): id  0 | task 100 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100000' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 100 |        eval time =   1000.00 ms /  100 tokens (   10.00 ms per token,    10.00 tokens per second)' },
    // slot 1: task 101 → должен получить СВОЙ LCP 50000
    { ts: T(4), msg: 'slot operator(): id  1 | task 101 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100000' },
    { ts: T(5), msg: 'slot print_timing: id  1 | task 101 |        eval time =   1000.00 ms /  100 tokens (   10.00 ms per token,    10.00 tokens per second)' }
  ]);
  // Последний таск (101, слот 1) — LCP слота 1
  assert.strictEqual(s.task.lcpTokens, 50000);
  // Таск 100 (в истории) — LCP слота 0, НЕ 50000 от слота 1
  const h100 = s.history.find(h => h.id === '100');
  assert.ok(h100, 'task 100 в истории');
  assert.strictEqual(h100.lcp, 90000);
});

// P1: выбор кандидата prompt-cache — upstream Pareto по f_keep/f_sim, а не max(lcp).
// Кандидат B имеет БОЛЬШЕ lcp (75000 > 70000), но НИЖЕ f_keep (0.750 < 0.875) →
// upstream не выберет B (нужно строгое улучшение ОБА f_keep и f_sim).
test('P1: кандидат prompt-cache выбирается по f_keep/f_sim (Pareto), не max lcp', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.001 (10/100000) > 0.100' },
    // кандидат A: cached=80000, lcp=70000, f_keep=0.875, f_sim=0.700
    { ts: T(1), msg: 'srv          load:    - prompt with length    80000, lcp =    70000, f_keep = 0.875, f_sim = 0.700' },
    // кандидат B: cached=100000, lcp=75000, f_keep=0.750, f_sim=0.750
    { ts: T(2), msg: 'srv          load:    - prompt with length    100000, lcp =    75000, f_keep = 0.750, f_sim = 0.750' },
    { ts: T(3), msg: 'srv          load:  - found better prompt with f_keep = 0.875, f_sim = 0.700' },
    { ts: T(4), msg: 'slot operator(): id  0 | task 200 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100000' }
  ]);
  // Выбран кандидат A (lcp 70000), не B (lcp 75000)
  assert.strictEqual(s.task.lcpTokens, 70000);
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram');
});

// P1: кандидат с f_keep < 0.25 отбрасывается (upstream guard). Тогда found better
// prompt идёт в fallback (f_sim из строки, tokens=0) → lcpTokens = оценка, не lcp кандидата.
test('P1: кандидат с f_keep < 0.25 отбрасывается (upstream guard)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.001 (10/100000) > 0.100' },
    // lcp=80000, но f_keep=0.100 < 0.25 → кандидат отброшен
    { ts: T(1), msg: 'srv          load:    - prompt with length    100000, lcp =    80000, f_keep = 0.100, f_sim = 0.900' },
    { ts: T(2), msg: 'srv          load:  - found better prompt with f_keep = 0.100, f_sim = 0.900' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 201 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100000' }
  ]);
  // Кандидат отброшен → fallback f_sim=0.900, tokens=0 → lcpTokens = round(100000*0.9)=90000,
  // а НЕ точный lcp отброшенного кандидата (80000)
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram');
  assert.strictEqual(s.task.lcpTokens, 90000);
});

// P1b: кандидат сравнивается с БАЗОВОЙ ЛИНИЕЙ текущего слота (upstream
// server-task.cpp), а не с предыдущим кандидатом. Контрпример:
//   base (текущий промпт слота): f_keep = 0.800, f_sim = 0.800
//   кандидат A: f_keep = 0.600, f_sim = 0.900  → 0.600 < 0.800 → REJECT
//   кандидат B: f_keep = 0.850, f_sim = 0.810  → оба > base → SELECT B
// Старый (кандидат-к-кандидату) код брал A (первый) и отбрасывал B
// (0.810 < 0.900) — неверно против upstream.
test('P1b: выбор RAM-кэша против базовой линии слота, не против кандидата', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.001 (10/100000) > 0.100' },
    // Базовая линия: текущий промпт слота (lcp 80000 из 100000 / 100000)
    { ts: T(1), msg: 'srv          load:  - looking for better prompt, base f_keep = 0.800, f_sim = 0.800' },
    // кандидат A: lcp=90000, f_keep=0.600 (< base 0.800) → отбрасывается,
    // НО f_sim=0.900 выше, чем у B — кандидат-к-кандидату ошибся бы здесь
    { ts: T(2), msg: 'srv          load:    - prompt with length   150000, lcp =    90000, f_keep = 0.600, f_sim = 0.900' },
    // кандидат B: lcp=81000, f_keep=0.850, f_sim=0.810 — оба > base → ВЫБРАН
    { ts: T(3), msg: 'srv          load:    - prompt with length    95294, lcp =    81000, f_keep = 0.850, f_sim = 0.810' },
    { ts: T(4), msg: 'srv          load:  - found better prompt with f_keep = 0.850, f_sim = 0.810' },
    { ts: T(5), msg: 'slot operator(): id  0 | task 202 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100000' }
  ]);
  // Выбран B (lcp 81000), не A (lcp 90000)
  assert.strictEqual(s.task.lcpTokens, 81000);
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram');
});

// P1c: пустой слот — базовая линия f_keep = -1.000 (upstream: "empty slot:
// any cache entry wins"): первый кандидат с f_keep >= 0.25 выигрывает,
// даже с низким f_sim.
test('P1c: пустой слот (base f_keep = -1.000) — любой кандидат с f_keep >= 0.25 выигрывает', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.000 (0/100000) > 0.100' },
    { ts: T(1), msg: 'srv          load:  - looking for better prompt, base f_keep = -1.000, f_sim = 0.000' },
    { ts: T(2), msg: 'srv          load:    - prompt with length    60000, lcp =    30000, f_keep = 0.500, f_sim = 0.300' },
    { ts: T(3), msg: 'srv          load:  - found better prompt with f_keep = 0.500, f_sim = 0.300' },
    { ts: T(4), msg: 'slot operator(): id  0 | task 203 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100000' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 30000);
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram');
});

// P2: per-task "acc per pos = (...)" (SPEC TOTAL, все реализации) парсится
// в task.specPosAcc (проценты) и приклеивается к таску из префикса строки.
test('P2: per-task "acc per pos" (upstream формат) парсится в specPosAcc', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 327 | draft acceptance = 0.612 (  25 accepted /   41 generated), mean len =  1.61' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 327 |      acc per pos = (0.910, 0.780, 0.620, 0.450, 0.300)' }
  ]);
  assert.deepStrictEqual(s.task.specPosAcc, ['91', '78', '62', '45', '30']);
  // Инвариант: mean = 1 + Σ rates = 1 + 3.06 = 4.06 ≠ 1.61 → false
  assert.strictEqual(s.task.specInvariantOk, false, 'инвариант нарушен (несогласованные данные)');
});

// P2b: согласованные данные → инвариант true (mean = 1 + Σ rates)
test('P2b: specInvariantOk — mean = 1 + Σ acc_rate_per_pos', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 328 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 328 | draft acceptance = 0.612 (  25 accepted /   41 generated), mean len =  4.06' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 328 |      acc per pos = (0.910, 0.780, 0.620, 0.450, 0.300)' }
  ]);
  assert.strictEqual(s.task.specInvariantOk, true, 'инвариант выполнен');
});

// #2: 36-rate инвариант на границе допуска: tol = 0.005 + 36*0.0005 = 0.023
test('P2c: 36-rate инвариант на границе допуска (tol = 0.023)', () => {
  const rates36 = Array(36).fill('0.100').join(', '); // Σ = 3.6 → 1 + Σ = 4.6
  const base = [
    { ts: T(0), msg: 'slot operator(): id  0 | task 340 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100' }
  ];
  // delta = |4.62 - 4.6| = 0.02 <= 0.023 → true
  const s1 = replay([...base,
    { ts: T(1), msg: 'slot print_timing: id  0 | task 340 | draft acceptance = 0.612 (  25 accepted /   41 generated), mean len =  4.62' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 340 |      acc per pos = (' + rates36 + ')' }
  ]);
  assert.strictEqual(s1.task.specInvariantOk, true, 'delta 0.02 <= tol 0.023');
  // delta = |4.63 - 4.6| = 0.03 > 0.023 → false
  const s2 = replay([...base,
    { ts: T(1), msg: 'slot print_timing: id  0 | task 340 | draft acceptance = 0.612 (  25 accepted /   41 generated), mean len =  4.63' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 340 |      acc per pos = (' + rates36 + ')' }
  ]);
  assert.strictEqual(s2.task.specInvariantOk, false, 'delta 0.03 > tol 0.023');
});

// P3: "statistics draft-mtp:" — SESSION CUMULATIVE, парсится в state.specMtp
// (НЕ в per-task поля). Формат common_speculative_print_stats.
test('P3: statistics draft-mtp → state.specMtp (session cumulative)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 329 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 329 | statistics       draft-mtp: #calls(b,g,a) =   12    12    12, #gen drafts =    60, #acc drafts =   48, #gen tokens =  300, #acc tokens =  248, #mean acc len = 3.07, #acc rate/pos = (1.000, 0.820, 0.550, 0.300, 0.100)' }
  ]);
  const sm = s.specMtp;
  assert.deepStrictEqual(sm.calls, [12, 12, 12], '#calls(b,g,a)');
  assert.strictEqual(sm.genDrafts, 60);
  assert.strictEqual(sm.accDrafts, 48);
  assert.strictEqual(sm.genTokens, 300);
  assert.strictEqual(sm.accTokens, 248);
  assert.strictEqual(sm.meanLen, 3.07);
  assert.deepStrictEqual(sm.posAcc, ['1.000', '0.820', '0.550', '0.300', '0.100']);
  // Инвариант: 1 + (1.000+0.820+0.550+0.300+0.100) = 3.77 ≠ 3.07 → false
  assert.strictEqual(sm.invariantOk, false);
  // Per-task поля НЕ заданы строкой источника (CRITICAL 3: не смешивать)
  assert.strictEqual(s.task.specAccepted, null, 'per-task spec не задан per-source строкой');
  // #1: specPosAcc — динамический массив (без фиксированных 5 позиций)
  assert.deepStrictEqual(s.task.specPosAcc, []);
});

// P4: "statistics ngram-mod:" — SESSION CUMULATIVE, парсится в state.ngram
test('P4: statistics ngram-mod → state.ngram (session cumulative)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 330 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 330 | statistics        ngram-mod: #calls(b,g,a) =  100   100   100, #gen drafts =   500, #acc drafts =  120, #gen tokens =  2500, #acc tokens =  1200, #mean acc len = 2.20, #acc rate/pos = (0.900, 0.300)' }
  ]);
  const ng = s.ngram;
  assert.deepStrictEqual(ng.calls, [100, 100, 100]);
  assert.strictEqual(ng.genDrafts, 500);
  assert.strictEqual(ng.accDrafts, 120);
  assert.strictEqual(ng.genTokens, 2500);
  assert.strictEqual(ng.accTokens, 1200);
  assert.strictEqual(ng.meanLen, 2.20);
  assert.deepStrictEqual(ng.posAcc, ['0.900', '0.300']);
  // Инвариант: 1 + (0.900+0.300) = 2.20 → true
  assert.strictEqual(ng.invariantOk, true);
});

test('V2: UI не имеет метки SOURCE=CHECKPOINT; источник и чекпоинт — отдельные элементы', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(!/SOURCE\s*=\s*CHECKPOINT/i.test(src), 'нет SOURCE=CHECKPOINT');
  assert.ok(src.includes("const SRC_LABEL = { 'hot-vram': 'HOT VRAM', 'prompt-cache-ram': 'RAM CACHE', 'empty': 'COLD' }"), 'SRC_LABEL без checkpoint');
  assert.ok(src.includes('✚CP'), 'чекпоинт — отдельный бейдж ✚CP');
});

// V3: splice чекпоинт-события по stage (не по тексту 'CACHED PREFIX')
test('V3: cached до restored — прематурный CACHED PREFIX вырезается, ровно одно KV_RESTORE (KV REUSE RAM)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7001 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7001 | cached n_tokens = 4800, memory_seq_rm [4800, end)' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7001 | restored context checkpoint (pos_min = 4799, pos_max = 4799, n_tokens = 4800, n_past = 4800, size = 270.000 MiB)' }
  ]);
  const evs = s.events.filter(e => e.stage === 'KV_RESTORE' && e.taskId === '7001');
  assert.strictEqual(evs.length, 1, 'ровно одно KV_RESTORE событие для таска');
  assert.strictEqual(evs[0].tag, 'KV REUSE RAM', 'осталось событие подъёма из RAM, CACHED PREFIX вырезан');
  assert.strictEqual(s.task.checkpointRestore, true);
  assert.strictEqual(s.task.kvRestoredTokens, 4800);
});

test('V3: инвариант checkpointRestore === true ⇒ kvRestoredTokens > 0', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7002 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 1000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7002 | restored context checkpoint (pos_min = 99, pos_max = 99, n_tokens = 100, n_past = 100, size = 5.600 MiB)' }
  ]);
  if (s.task.checkpointRestore) {
    assert.ok(s.task.kvRestoredTokens > 0, 'checkpointRestore ⇒ kvRestoredTokens > 0');
  }
  assert.strictEqual(s.task.checkpointRestore, true);
  assert.strictEqual(s.task.kvRestoredTokens, 100);
});

// V4: частичный LCP (обрыв внутри чекпоинта): n_past < n_tokens.
// kvRestoredTokens = n_past (фактически применённые в слот), а RAM-кэш
// ищет по n_tokens (размер снапшота — ключ из строки created).
test('V4: n_past < n_tokens — kvRestoredTokens = n_past, RAM-кэш пульсирует по n_tokens', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 7003 | created context checkpoint 1 of 32 (pos_min = 0, pos_max = 4999, n_tokens = 5000, size = 270.000 MiB)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7003 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5200' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7003 | restored context checkpoint (pos_min = 0, pos_max = 4999, n_tokens = 5000, n_past = 4200, size = 270.000 MiB)' }
  ]);
  assert.strictEqual(s.task.kvRestoredTokens, 4200, 'n_past — токены, фактически применённые в слот');
  assert.strictEqual(s.task.checkpointRestore, true);
  // Ключ RAM-кэша = (pos_min, pos_max) = (0, 4999), НЕ n_past (4200)
  assert.strictEqual(s.ramCache.slots['0:0:4999'].restoreCount, 1);
});

// V5: «0 prompts» = пустой PROMPT-кэш, НЕ «нет чекпоинтов» — стойка не сбрасывается
test('V5: cache state 0 prompts при живом чекпоинте — стойка чекпоинтов не сбрасывается', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'srv update: - cache state: 0 prompts, 0.000 MiB (limits: 40960.000 MiB, 123136 tokens, 42949672960 est)' }
  ]);
  assert.strictEqual(s.ramCache.promptsCount, 0);
  assert.ok(s.ramCache.slots['0:100:100'], 'слот чекпоинта жив после «0 prompts»');
  assert.strictEqual(s.ramCache.cpCurrent, 1);
  assert.strictEqual(s.ramCache.cpUsedMiB, 1.5);
});

// V6: --parallel > 1 — чекпоинты владение пер-слот (upstream: slot.prompt.checkpoints).
// N в "created context checkpoint N of M" — позиция в списке ЭТОГО слота,
// поэтому resync НЕ агрегирует N-of-M глобально: первое создание slot1
// не должно вытеснять живой чекпоинт slot0.
test('V6: parallel=2 — чекпоинт slot1 не вытесняет slot0 (per-slot N-of-M)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  1 | task 328 | created context checkpoint 1 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' }
  ]);
  assert.ok(s.ramCache.slots['0:100:100'], 'чекпоинт slot0 жив');
  assert.ok(s.ramCache.slots['1:200:200'], 'чекпоинт slot1 жив');
  assert.strictEqual(s.ramCache.slots['0:100:100'].removedAt, undefined);
  assert.strictEqual(s.ramCache.slots['1:200:200'].removedAt, undefined);
  assert.strictEqual(s.ramCache.slots['0:100:100'].slot, 0);
  assert.strictEqual(s.ramCache.slots['1:200:200'].slot, 1);
  assert.strictEqual(s.ramCache.cpCurrent, 2);
  assert.strictEqual(s.ramCache.cpUsedMiB, 3.5);
  // Кольцо 2: метки несут владение слотом
  const model = ringModel(s);
  assert.deepStrictEqual(model.cp.marks.map(m => m.slot), [0, 1]);
});

test('V6: per-slot erased — erased id 1 стирает только slot1, slot0 не тронут', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  1 | task 328 | created context checkpoint 1 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' },
    { ts: T(2), msg: 'slot operator(): id  1 | task 328 | erased invalidated context checkpoint (pos_min = 200, pos_max = 200, n_tokens = 201, n_swa = 0, pos_next = 0, size = 2.000 MiB)' }
  ]);
  assert.strictEqual(s.ramCache.slots['1:200:200'].removeReason, 'invalidated');
  assert.strictEqual(s.ramCache.slots['0:100:100'].removedAt, undefined);
  assert.strictEqual(s.ramCache.slots['0:100:100'].evictedAt, undefined);
  assert.strictEqual(s.ramCache.cpCurrent, 1);
  assert.strictEqual(s.ramCache.cpUsedMiB, 1.5);
});

test('V6: per-slot N==1 resync — лишние slot0 снимаются, slot1 не тронут', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 32 (pos_min = 200, pos_max = 200, n_tokens = 201, size = 2.000 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  1 | task 328 | created context checkpoint 1 of 32 (pos_min = 300, pos_max = 300, n_tokens = 301, size = 2.500 MiB)' },
    { ts: T(3), msg: 'slot create_check: id  0 | task 329 | created context checkpoint 1 of 32 (pos_min = 400, pos_max = 400, n_tokens = 401, size = 3.000 MiB)' }
  ]);
  // slot0: N=1 → старейшие 101 и 201 сняты (resync), жив 401
  assert.strictEqual(s.ramCache.slots['0:100:100'].removeReason, 'resync');
  assert.strictEqual(s.ramCache.slots['0:200:200'].removeReason, 'resync');
  assert.strictEqual(s.ramCache.slots['0:400:400'].removedAt, undefined);
  // slot1: N=1, один живой — глобальная агрегация сняла бы и его
  assert.strictEqual(s.ramCache.slots['1:300:300'].removedAt, undefined);
  assert.strictEqual(s.ramCache.cpCurrent, 2);
  assert.strictEqual(s.ramCache.cpUsedMiB, 5.5);
});

// V1 cross-check: n_ctx из разных источников обязан совпадать; первый источник
// побеждает, расхождение фиксируется один раз (флаг + одно событие)
test('cross-check: init vs prompt disagree -> first wins, mismatch fixed once', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 99999, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.n_ctx, 123136, 'первый источник побеждает');
  assert.strictEqual(s.n_ctxSrc, 'init');
  assert.deepStrictEqual(s.n_ctxMismatch, { init: 123136, prompt: 99999 });
  const evs = s.events.filter(e => e.tag === 'NCTX_MISMATCH');
  assert.strictEqual(evs.length, 1, 'ровно одно mismatch-событие');
  assert.strictEqual(evs[0].stage, 'SYS');
});

test('cross-check: init и prompt совпадают -> mismatch нет', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.n_ctx, 123136);
  assert.strictEqual(s.n_ctxSrc, 'init');
  assert.strictEqual(s.n_ctxMismatch, null);
  assert.strictEqual(s.events.filter(e => e.tag === 'NCTX_MISMATCH').length, 0);
});

test('cross-check: повторное расхождение -> всё ещё одно событие, первое расхождение сохранено', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 99999, n_keep = 0, task.n_tokens = 100' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 328 | new prompt, n_ctx_slot = 88888, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.deepStrictEqual(s.n_ctxMismatch, { init: 123136, prompt: 99999 });
  assert.strictEqual(s.events.filter(e => e.tag === 'NCTX_MISMATCH').length, 1);
});

test('cross-check: новый initializing (рестарт сервера) сбрасывает cross-check — ложного mismatch нет', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 128000, kv_unified = 'false'" },
    { ts: T(2), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.n_ctx, 128000, 'n_ctx последней жизни');
  assert.strictEqual(s.n_ctxSrc, 'init');
  assert.strictEqual(s.n_ctxMismatch, null, 'разные жизни — не расхождение');
  assert.strictEqual(s.events.filter(e => e.tag === 'NCTX_MISMATCH').length, 0);
});

// ===== M1: /metrics (llama.cpp Prometheus) =====

test('M1: parsePrometheusMetrics — 3 счётчика, комментарии/HELP/TYPE, label-скобки', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/function parsePrometheusMetrics\(text\) \{[\s\S]*?\n\}/);
  assert.ok(m, 'parsePrometheusMetrics found in www.js');
  const fake = { serverMetrics: { queueDeferred: 0, promptTokensTotal: 0, tokensPredictedTotal: 0, fetchedAt: 0 } };
  const parse = new Function('state', m[0] + '\nreturn parsePrometheusMetrics;')(fake);
  parse([
    '# HELP llamacpp:requests_deferred Deferred requests',
    '# TYPE llamacpp:requests_deferred gauge',
    'llamacpp:requests_deferred 2',
    'llamacpp:kv_cache_usage_ratio{slot="0"} 0.42',
    'llamacpp:prompt_tokens_total 12345',
    'llamacpp:tokens_predicted_total 678',
    'llamacpp:server_load 0.1'
  ].join('\n'));
  const sm = fake.serverMetrics;
  assert.strictEqual(sm.queueDeferred, 2);
  // kv_cache_usage_ratio НЕ экспортируется tools/server — строка игнорируется
  assert.strictEqual(sm.kvAuthoritativeRatio, undefined, 'kv_cache_usage_ratio игнорируется');
  assert.strictEqual(sm.promptTokensTotal, 12345);
  assert.strictEqual(sm.tokensPredictedTotal, 678);
  assert.ok(sm.fetchedAt > 0, 'fetchedAt обновлён');
});

test('M1: parsePrometheusMetrics — CRLF и повторный вызов перезаписывает значения', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/function parsePrometheusMetrics\(text\) \{[\s\S]*?\n\}/);
  const fake = { serverMetrics: { queueDeferred: 7, promptTokensTotal: 0, tokensPredictedTotal: 0, fetchedAt: 0 } };
  const parse = new Function('state', m[0] + '\nreturn parsePrometheusMetrics;')(fake);
  parse('llamacpp:requests_deferred 3\r\nllamacpp:kv_cache_usage_ratio 0.9\r\n');
  assert.strictEqual(fake.serverMetrics.queueDeferred, 3, 'CRLF не ломает значение');
  assert.strictEqual(fake.serverMetrics.kvAuthoritativeRatio, undefined, 'kv_cache_usage_ratio игнорируется');
});

test('M1: /metrics — отдельный pollMetrics: свой интервал 2с, свой busy-флаг, не в pollServer', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/async function pollMetrics\(\) \{[\s\S]*?\n\}/);
  assert.ok(m, 'pollMetrics found');
  assert.ok(m[0].includes('${SERVER_URL}/metrics'), 'fetch /metrics есть');
  assert.ok(m[0].includes('metricsPollBusy'), 'свой busy-флаг');
  assert.ok(m[0].includes('new AbortController()'), 'свой AbortController');
  const ps = src.match(/async function pollServer\(\) \{[\s\S]*?\n\}/);
  assert.ok(ps, 'pollServer found');
  assert.ok(!ps[0].includes('/metrics'), '/metrics вынесен из pollServer');
  assert.ok(src.includes('setInterval(pollMetrics, 2000)'), 'интервал 2с');
});

test('M1: replay не ходит в /metrics — serverMetrics нулевой и детерминированный', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" }
  ]);
  assert.deepStrictEqual(s.serverMetrics, {
    queueDeferred: 0,
    promptTokensTotal: 0, tokensPredictedTotal: 0, fetchedAt: 0,
    metricsSeen: false // #18: /metrics ещё не ответил — нули не данные
  });
});

test('M1: UI — тулбар плеера ЕСТЬ в HTML_PAGE (#4); offscreen-mermaid на месте; JS гвардит элементы', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(!src.includes('class="ctl-bar"'), 'старый .ctl-bar не в HTML_PAGE');
  assert.ok(src.includes('class="player-toolbar"'), 'тулбар плеера в HTML_PAGE (#4)');
  assert.ok(src.includes('id="ctl-play"'), 'кнопка play/pause в HTML_PAGE');
  assert.ok(src.includes('id="ctl-restart"'), 'кнопка restart в HTML_PAGE');
  assert.ok(src.includes('id="ctl-speed"'), 'селект скорости в HTML_PAGE');
  assert.ok(src.includes('id="ctl-file"'), 'загрузка лога в HTML_PAGE');
  assert.ok(!src.includes('class="status-compact-wrap"'), 'компактный статус не в HTML_PAGE');
  assert.ok(src.includes('position:fixed; left:-9999px'), 'offscreen-контейнер mermaid сохранён');
  assert.ok(src.includes('if (qb) {'), 'JS безопасно гвардит бейдж');
});

// ===== M3: KV ROI без магических чисел =====

test('M3: lastPrefillTps — финальное измерение из prompt eval time', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'prompt eval time = 1234.56 ms / 100 tokens (12.35 ms per token, 81.00 tokens per second)' }
  ]);
  assert.strictEqual(s.lastPrefillTps, 81.0);
});

test('M3: lastPrefillTps — промежуточное (progress) переопределяется финальным (eval time)', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | prompt processing, n_tokens = 500, progress = 0.50, 100.00 tokens per second' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 327 | prompt eval time = 1234.56 ms / 1000 tokens (1.23 ms per token, 81.00 tokens per second)' }
  ]);
  assert.strictEqual(s.lastPrefillTps, 81.0, 'финальное измерение побеждает');
});

test('M3: без измеренного префилла lastPrefillTps = 0 (ROI не выдумывает скорость)', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.lastPrefillTps, 0);
});

test('M3: UI — ROI-строка в легенде, без магической константы 2500 t/s', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('id="row-kv-roi"'), 'ROI-строка в HTML');
  assert.ok(src.includes('id="lbl-kv-roi"'), 'ROI-значение в HTML');
  assert.ok(!src.includes('2500'), 'нет магического 2500 t/s');
  assert.ok(src.includes('s.lastPrefillTps'), 'ROI берёт измеренную скорость');
});

test('M3: layout-shift — ROI-строка всегда в DOM (серая), live-строка без border-left', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('id="row-kv-roi" style="background:transparent; border:none; padding:1px 0;"'), 'ROI-строка без display:none');
  assert.ok(src.includes('<div class="legend-title" style="color:var(--dim)"><span data-i18n-title="saved_time_title">SAVED TIME:</span></div>'), 'SAVED TIME-заголовок серый (i18n)');
  assert.ok(src.includes('id="lbl-kv-roi" style="color:var(--dim); font-size:11px"'), 'ROI-значение серое');
  assert.ok(!src.includes("roiRow.style.display"), 'renderState не прячет ROI-строку');
  assert.ok(src.includes('.hist-row-live{background:rgba(56,189,248,0.08) !important}'), 'live-строка без border-left');
  assert.ok(src.includes('.hist-row-live td:first-child{box-shadow:inset 3px 0 0 var(--input) !important}'), 'синяя полоска через inset box-shadow');
});

// ===== M4: bottleneck-бейдж (чисто UI) =====

test('M4: UI — bottleneck: честные формулировки фаз (без претензии на замер), IDLE в простое', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('id="core-bottleneck"'), 'бейдж в HTML');
  assert.ok(src.includes('PREFILL / BATCH COMPUTE'), 'PREFILL → batch compute');
  assert.ok(src.includes('DECODE / TOKEN STEP'), 'DECODE → token step');
  assert.ok(!src.includes('COMPUTE-BOUND'), 'нет ложной претензии на хардварный замер');
  const m = src.match(/const bn = document\.getElementById\('core-bottleneck'\);[\s\S]*?else bn\.innerText = 'IDLE';/);
  assert.ok(m, 'логика отображения в renderState');
});

// ===== M5: fast-path декода (чистый reorder) =====

test('M5: ветка декода стоит ПЕРЕД ветками 1–8 (fast path), после initializing', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  const iInit = src.indexOf("line.includes('initializing,')");
  const iDecode = src.indexOf("line.includes('tg =')");
  const iRam = src.indexOf("line.includes('found better prompt')");
  assert.ok(iInit > 0 && iDecode > 0 && iRam > 0, 'все три маркера найдены');
  assert.ok(iInit < iDecode, 'fast path после initializing');
  assert.ok(iDecode < iRam, 'fast path перед веткой 1 (Host RAM hit)');
});

// ===== M5b: return в конце fast-path =====

test('M5b: fast-path декода завершается return — строка не пробегает остаток цепочки', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/if \(line\.includes\('tg ='\) && \(line\.includes\('n_gen ='\) \|\| line\.includes\('n_decoded ='\)\)\) \{[\s\S]*?\n  \}/);
  assert.ok(m, 'fast-path блок найден');
  assert.ok(m[0].includes('return; // M5b'), 'return в конце блока');
});

// ===== C1: унификация цветовой палитры (UI + Canvas) =====

test('C1: :root — семантические переменные, старые бирюзы удалены', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  for (const v of ['--prefix:#6366f1', '--slot:#94a3b8', '--kv-save:#14b8a6', '--kv-evict:#475569', '--mtp:#06b6d4']) {
    assert.ok(src.includes(v), 'есть ' + v);
  }
  assert.ok(!src.includes('#39c9cf'), 'нет хардкода #39c9cf');
  assert.ok(!src.includes('#39c5cf'), 'нет старого --prefix #39c5cf');
  assert.ok(!src.includes('rgba(57,197,207') && !src.includes('rgba(57,201,207'), 'нет бирюзовых rgba-фонов');
});

test('C1: плашки лога — цветные (откат монохрома), текст лога спокойный серый', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('.log-tag { width: 104px; min-width: 104px; height: 22px; flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; box-sizing: border-box; font-size: 9.5px; padding: 0 4px; border-radius: 4px; font-weight: bold; letter-spacing: 0.2px; margin-top: 1px; }'), 'log-tag базовый размер без монохром-оверрайда');
  assert.ok(src.includes('.log-row { margin: 2px 0; padding: 4px 8px; border-radius: 4px; background: #070b12; border-left: 3px solid var(--border);'), 'log-row базовый бордюр var(--border)');
  assert.ok(src.includes('.log-msg { flex: 1; min-width: 0; word-break: break-word; color: var(--dim) !important; }'), 'текст лога — спокойный серый');
  assert.ok(src.includes('.log-msg b { color: var(--dim) !important; font-weight: 600; }'), 'цифры в логе — строго серый var(--dim)');
  assert.ok(src.includes('.log-row.MTP { border-left-color: var(--mtp); }'), 'цветной MTP-бордюр возвращён');
  assert.ok(src.includes('.log-row.DONE { border-left-color: #64748b; }'), 'цветной DONE-бордюр возвращён');
  assert.ok(src.includes('.log-row.PREFILL .log-tag'), 'цветные PREFILL-плашки возвращены');
  assert.ok(!src.includes('background: #0f172a !important'), 'монохромный оверрайд удалён');
});

test('C1: Canvas — ring2 seafoam/coral, ring3 coral, луч-градиент', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes("ctx.strokeStyle = touch > 0 ? '#22c55e' : '#14b8a6';"), 'ring2 живая метка → #14b8a6');
  assert.ok(src.includes("ctx.strokeStyle = '#64748b';"), 'ring2 смерть → серый сланец #64748b');
  assert.ok(src.includes("ctx.strokeStyle = '#f43f5e';"), 'ring3 смерть → #f43f5e');
  assert.ok(src.includes('const grad = ctx.createLinearGradient('), 'луч переноса — градиент');
  assert.ok(src.includes("grad.addColorStop(0, 'rgba(20, 184, 166, ' + (ba * 0.9) + ')');"), 'градиент старт Host RAM');
});

// ===== C2: «Спокойная спекуляция» — MTP циан + полупрозрачный эквалайзер =====

test('C2: :root — --mtp циан, --mtp-dim акцент, фуксия удалена', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('--mtp:#06b6d4'), '--mtp циан');
  assert.ok(src.includes('--mtp-dim:#22d3ee'), '--mtp-dim светлый акцент');
  assert.ok(!src.includes('ec4899') && !src.includes('236,72,153'), 'фуксия нигде не осталась');
});

test('C2: эквалайзер — полупрозрачные столбики с засечкой, MTP в таблице --mtp, вторичные метрики легенды приглушены', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('background:rgba(6,182,212,0.25);border-top:2px solid var(--mtp-dim)'), 'mtp-fill: полупрозрачное тело + засечка');
  assert.ok(src.includes('id="lbl-ngram" style="color:var(--dim)'), 'N-GRAM приглушён → --dim');
  assert.ok(src.includes('<th style="color:var(--mtp)" data-i18n-title="th_spec_title">SPEC</th>'), 'история th SPEC → --mtp (i18n)');
  assert.ok(src.includes('<td style="color:var(--mtp);font-size:10px">'), 'история td MTP → --mtp + 10px');
  assert.ok(src.includes('id="lbl-mtp-step" data-i18n="tok_per_step" style="font-size:10.5px; color:var(--dim)'), 'lbl-mtp-step приглушён → --dim (i18n)');
});

test('C3: SPEED объединена в DECODE — 11 колонок, <th>SPEED удалён', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(!src.includes('<th>SPEED'), 'th SPEED удалён');
  const thead = src.match(/<thead>[\s\S]*?<\/thead>/);
  assert.ok(thead, 'thead есть');
  assert.strictEqual((thead[0].match(/<th[\s>]/g) || []).length, 11, 'ровно 11 колонок');
  assert.ok(src.includes("'  <td style=\"color:var(--decode)\">' + num(h.decode).toLocaleString() + decodeSpeedStr + '</td>'"), 'история: DECODE + скорость в скобках');
  assert.ok(src.includes("'  <td style=\"color:var(--decode)\">' + num(t.generatedTokens).toLocaleString() + liveDecodeSpeed + '</td>'"), 'live: DECODE + текущая скорость');
  assert.ok(src.includes('const decodeSpeedStr = h.cancelled'), 'история: cancelled → (✗)');
  assert.ok(!src.includes('liveSpeedHtml'), 'liveSpeedHtml удалён');
});

// ===== E1–E6: телеметрия старта + цвета чекпоинтов =====

test('E1: INPUT регистрируется до initNewTask — лента снизу вверх: INPUT → PREFIX → SLOT SELECT', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7001 | checking sim = 0.960 (4800/5000)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7001 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' }
  ]);
  const idx = tag => s.events.findIndex(e => e.taskId === '7001' && e.tag === tag);
  assert.ok(idx('INPUT') !== -1, 'INPUT есть');
  assert.ok(idx('PREFIX') !== -1, 'PREFIX есть');
  assert.ok(idx('SLOT SELECT') !== -1, 'SLOT SELECT есть');
  assert.ok(idx('SLOT SELECT') < idx('PREFIX'), 'SLOT SELECT новее PREFIX (выше в ленте)');
  assert.ok(idx('PREFIX') < idx('INPUT'), 'PREFIX новее INPUT (INPUT лежит внизу)');
});

test('E2/E3: restored до cached — один «Источник:» в SLOT SELECT, близнец CACHED PREFIX не создаётся', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7010 | checking sim = 0.960 (4800/5000)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7010 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7010 | restored context checkpoint (pos_min = 4799, pos_max = 4799, n_tokens = 4800, n_past = 4800, size = 270.000 MiB)' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 7010 | cached n_tokens = 4800, memory_seq_rm [4800, end)' }
  ]);
  const evs = s.events.filter(e => e.stage === 'KV_RESTORE' && e.taskId === '7010');
  assert.strictEqual(evs.length, 1, 'ровно одно KV_RESTORE');
  assert.strictEqual(evs[0].tag, 'KV REUSE RAM', 'подъём из RAM, без близнеца CACHED PREFIX');
  const sel = s.events.find(e => e.stage === 'PREFIX' && e.tag === 'SLOT SELECT' && e.taskId === '7010');
  assert.ok(sel, 'SLOT SELECT есть');
  assert.strictEqual(sel.params.sourceKey, 'src_checkpoint', 'источник — ровно один, checkpoint (Host RAM → VRAM)');
});

test('E4: KV EVICT RAM — сланец #475569 (текст #94a3b8); ring2 смерть #64748b, ring3 смерть #f43f5e', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('--kv-evict:#475569'), '--kv-evict сланец');
  assert.ok(src.includes('background: rgba(71,85,105,0.2); color: #94a3b8;'), 'плашка EVICT: фон + текст сланец');
  assert.ok(src.includes("ctx.strokeStyle = '#64748b';"), 'ring2 смерть → #64748b (canvas, отдельно от лога)');
  assert.ok(src.includes("ctx.strokeStyle = '#f43f5e';"), 'ring3 смерть → #f43f5e');
  assert.ok(!src.includes('#f87171'), 'старый лососевый #f87171 удалён');
  assert.ok(!src.includes('#fb7185'), 'старый красный текст EVICT удалён');
});

test('E5: ✚CP (val-cached) → var(--kv-save); история — семантические статусы чекпоинта', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes("color:var(--kv-save);font-size:10px\" title=\"' + window.__wwwT('tt_cp_restored_vram')"), 'val-cached ✚CP → --kv-save (i18n title)');
  // История больше не использует «✚CP» — вместо него семантические статусы:
  // HOT+CP ↺ (откат к чекпоинту внутри VRAM) и RAM→VRAM (восстановление из Host RAM)
  assert.ok(src.includes('status-rewind'), 'история: HOT+CP ↺ (откат к чекпоинту в VRAM)');
  assert.ok(src.includes('status-ram'), 'история: RAM→VRAM / RAM CACHE (восстановление из Host RAM)');
});

test('E6: лента контекста (tape) удалена — HTML и JS без следов', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(!src.includes('tape-bar'), 'tape-bar удалён');
  assert.ok(!src.includes('tape-ticks'), 'tape-ticks удалён');
  assert.ok(!src.includes('lbl-tape-'), 'lbl-tape-* удалены');
  assert.ok(!src.includes('lastTicksHtml'), 'lastTicksHtml удалён');
});

// --- ТЗ: C++-фиксы llama.cpp + DONE slate + оптимизация лога -----------------

test('F1: LRU — "selected slot by LRU" → серый [SLOT SELECT] Cold Start, slotSource=empty', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7020 | checking sim = 0.050 (250/5000)' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7020 | selected slot by LRU' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7020 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' }
  ]);
  const sel = s.events.find(e => e.stage === 'PREFIX' && e.tag === 'SLOT SELECT' && e.taskId === '7020');
  assert.ok(sel, 'SLOT SELECT (LRU) есть');
  assert.strictEqual(sel.detailKey, 'ev_slot_lru_detail', 'Cold Start в detail-ключе');
  assert.strictEqual(sel.params.slot, 0, 'слот #0 в params');
  assert.strictEqual(s.task.slotSource, 'empty', 'slotSource = empty');
  assert.strictEqual(s.task.lcpTokens, 0, 'LCP нет');
});

test('F2: INPUT — currentStage фиксируется сразу после new prompt (без мигания IDLE)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7030 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' }
  ]);
  assert.strictEqual(s.currentStage, 'INPUT');
});

test('F3: found better prompt без кандидата — f_sim из самой строки (RAM fallback)', () => {
  const s = replay([
    { ts: T(0), msg: 'I srv          load:  - found better prompt with f_keep = 1.000, f_sim = 0.999' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7040 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' }
  ]);
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram', 'provenance — RAM cache');
  assert.strictEqual(s.task.lcpRatio, 0.999, 'ratio из f_sim строки');
});

// F4 удалён: строка "prompt done" не существует в tools/server (dead code,
// третий VERDICT fix 5) — ветка и её тест убраны.

test('F5: stop processing — finalSlotTokens из n_tokens', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7060 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7060 | eval time = 1000.000 ms / 100 tokens, with 100.000 tokens per second' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7060 | stop processing: n_tokens = 5100, truncated = 0' }
  ]);
  assert.strictEqual(s.task.finalSlotTokens, 5100);
});

test('F6: UI — log-time скрыт по умолчанию, tooltip справа от курсора, сворачиваемый EVENT LOG', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('.phase-DONE { box-shadow: 0 0 0 1px #64748b, 0 0 12px rgba(100,116,139,0.25); color: #94a3b8; }'), 'phase-DONE сланец');
  assert.ok(src.includes('<span class="log-time">'), 'log-time в DOM (управляется CSS)');
  assert.ok(src.includes('.log-row .log-time { display: none;'), 'CSS .log-time скрыт по умолчанию');
  assert.ok(src.includes('ev.clientX - rect.left + 18'), 'tooltip строго справа от курсора (+18)');
  assert.ok(src.includes('id="btn-toggle-log"'), 'кнопка сворачивания EVENT LOG');
  assert.ok(src.includes('.main-layout.log-collapsed { grid-template-columns: 1fr 36px; }'), 'свёрнутая колонка 36px');
});

test('F7: UI — фильтр логов: поповер, галочки категорий, тумблер времени, localStorage', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('id="btn-log-filter"'), 'кнопка 🌪 Фильтр');
  assert.ok(src.includes('id="log-filter-popover"'), 'поповер фильтров');
  assert.ok(src.includes('id="filter-show-time"'), 'тумблер времени');
  assert.ok(src.includes('id="btn-filter-reset"'), 'кнопка сброса «Все»');
  assert.ok(src.includes('#log-feed.show-time .log-row .log-time { display: inline-block; }'), 'show-time включает время');
  assert.ok(src.includes('#log-feed.hide-PREFILL .log-row.PREFILL { display: none !important; }'), 'hide-PREFILL');
  assert.ok(src.includes('#log-feed.hide-PREFIX .log-row.PREFIX:not([data-tag="SLOT SELECT"]) { display: none !important; }'), 'hide-PREFIX не трогает SLOT SELECT');
  assert.ok(src.includes('#log-feed.hide-SLOT_SELECT .log-row[data-tag="SLOT SELECT"] { display: none !important; }'), 'hide-SLOT_SELECT');
  assert.ok(src.includes('#log-feed.hide-DONE .log-row.DONE { display: none !important; }'), 'hide-DONE');
  assert.ok(src.includes("localStorage.setItem('monitor_show_time'"), 'localStorage: время');
  assert.ok(src.includes("'monitor_filter_' + target"), 'localStorage: категории');
  const targets = ['INPUT', 'PREFIX', 'SLOT_SELECT', 'KV_RESTORE', 'KV_RAM', 'PREFILL', 'DECODE', 'MTP', 'DONE'];
  for (const t of targets) {
    assert.ok(src.includes('class="tag-filter" data-target="' + t + '"'), 'галочка категории ' + t);
    assert.ok(src.includes('#log-feed.hide-' + t + ' .log-row'), 'CSS hide-' + t);
  }
});

// ===== H: точечный hardening стейт-машины (P0/P1) =====

test('H1: stop processing чужого таска не трогает текущее состояние (isForeignTask)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7060 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7061 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 1000' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7060 | stop processing: n_tokens = 5100, truncated = 1' }
  ]);
  assert.strictEqual(s.task.isTruncated, false, 'чужой truncated=1 не помечает текущий таск');
  assert.strictEqual(s.task.finalSlotTokens, undefined, 'чужой n_tokens не зальёт finalSlotTokens');
});

test('H2: draft acceptance отставшего таска приклеивается к СВОЕМУ таску в истории', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7060 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7060 | eval time = 1000.000 ms / 100 tokens, with 100.000 tokens per second' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7061 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 1000' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 7060 | draft acceptance = 0.850 (17 accepted / 20 generated), mean len = 1.850' }
  ]);
  const h = (s.history || []).find(x => String(x.id) === '7060');
  assert.ok(h, 'запись таска 7060 в истории');
  assert.strictEqual(h.specAccepted, 17, 'SPEC-поля приклеены к своему таску');
  assert.strictEqual(h.specGenerated, 20);
  assert.strictEqual(h.specAcceptance, '85.0');
  assert.strictEqual(h.specMeanLen, 1.85);
  assert.strictEqual(s.task.specAccepted, null, 'текущий таск 7061 не получил чужие SPEC-поля');
  assert.strictEqual(s.task.specGenerated, null);
  const ev = s.events.find(e => e.tag === 'MTP');
  assert.ok(ev, 'MTP-эвент создан');
  assert.strictEqual(ev.taskId, '7060', 'эвент приписан таску из строки');
});

test('H3: ring model — n_ctxMismatch → fail-closed, ось кольца 2 аннулирована (frac = null)', () => {
  const s = replay([
    { ts: T(0), msg: "srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'" },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 99999, n_keep = 0, task.n_tokens = 100' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' }
  ]);
  assert.ok(s.n_ctxMismatch, 'mismatch зафиксирован');
  const model = ringModel(s);
  assert.strictEqual(model.cp.state, 'ok');
  assert.ok(model.cp.marks.length > 0, 'марки чекпоинтов есть');
  for (const mk of model.cp.marks) {
    assert.strictEqual(mk.frac, null, 'ось аннулирована при mismatch');
  }
});

// ===== L: диагностика REQUEST HISTORY (LCP-DEBUG: точный LCP + DIVERGE + CP SKEW) =====

test('L1: [LCP-DEBUG] сводка — точный LCP переопределяет оценку из "checking sim"', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.641 (16000/25455) > 0.100' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 4492 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 25455' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 4492 | [LCP-DEBUG] slot_tokens=25455 input_tokens=25455 lcp=16328' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 16328, 'точный LCP из сводной строки, не 16000 из checking sim');
  assert.strictEqual(s.task.deltaTokens, 25455 - 16328, 'delta пересчитан от точного LCP');
});

test('L2: DIVERGE — точная точка расхождения в таске и в истории', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 4492 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 25455' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 4492 | [LCP-DEBUG] i=16328  A=8839     Note                     B=20320    Called                   <== DIVERGE' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 4492 | prompt eval time =  10000.00 ms / 9127 tokens (    1.09 ms per token,   912.70 tokens per second)' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 4492 | n_gen =    101, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(4), msg: 'slot print_timing: id  0 | task 4492 |        eval time =   99864.69 ms /  2574 tokens (   38.81 ms per token,    25.76 tokens per second)' }
  ]);
  const div = { pos: 16328, tokA: '8839', pieceA: 'Note', tokB: '20320', pieceB: 'Called' };
  assert.deepStrictEqual(s.task.divergeInfo, div, 'divergeInfo у текущего таска');
  const h = s.history[0];
  assert.ok(h, 'таск в истории');
  assert.deepStrictEqual(h.divergeInfo, div, 'divergeInfo перенесён в запись истории');
});

test('L3: DIVERGE чужого таска игнорируется (guard по task-id)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 100 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 99 | [LCP-DEBUG] i=100  A=1111     old                      B=2222    new                    <== DIVERGE' }
  ]);
  assert.strictEqual(s.task.divergeInfo, null, 'DIVERGE таска 99 не приклеился к таску 100');
});

test('L4: построчные [LCP-DEBUG] без DIVERGE не меняют состояние', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 4492 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 25455' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 4492 | [LCP-DEBUG] i=100  A=1234     hello                    B=1234    hello' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 0, 'построчная строка — не сводка, LCP не задан');
  assert.strictEqual(s.task.divergeInfo, null, 'нет маркера DIVERGE — расхождения нет');
});

test('L5: CP SKEW = LCP − восстановленный префикс (переплата за шаг чекпоинта)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.997 (98819/99071) > 0.100' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 5000 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 99071' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 5000 | restored context checkpoint (pos_min = 24085, pos_max = 24085, n_tokens = 24086, n_past = 24086, size = 140.000 MiB)' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 5000 | prompt eval time =  10000.00 ms / 74733 tokens (    0.13 ms per token,  7473.30 tokens per second)' },
    { ts: T(4), msg: 'slot print_timing: id  0 | task 5000 | n_gen =    101, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(5), msg: 'slot print_timing: id  0 | task 5000 |        eval time =   99864.69 ms /  2574 tokens (   38.81 ms per token,    25.76 tokens per second)' }
  ]);
  const h = s.history[0];
  assert.strictEqual(h.lcp, 98819, 'LCP из checking sim');
  assert.strictEqual(h.kvReuse, 24086, 'восстановлено из чекпоинта');
  assert.strictEqual(h.cpSkew, 98819 - 24086, 'skew = LCP − restored');
});

test('L6: без LCP-данных skew не отрицательный (clamp в 0)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 5001 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 5001 | cached n_tokens = 4800, memory_seq_rm [4800, end)' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 5001 | restored context checkpoint (pos_min = 4799, pos_max = 4799, n_tokens = 4800, n_past = 4800, size = 270.000 MiB)' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 5001 | prompt eval time =  1000.00 ms / 200 tokens (    5.00 ms per token,   200.00 tokens per second)' },
    { ts: T(4), msg: 'slot print_timing: id  0 | task 5001 | n_gen =    101, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(5), msg: 'slot print_timing: id  0 | task 5001 |        eval time =   99864.69 ms /  2574 tokens (   38.81 ms per token,    25.76 tokens per second)' }
  ]);
  const h = s.history[0];
  assert.strictEqual(h.lcp, 0, 'нет LCP-данных — cold start');
  assert.strictEqual(h.kvReuse, 4800);
  assert.strictEqual(h.cpSkew, 0, 'skew не уходит в минус');
});

test('L7: forcing full re-processing — LCP-совпадение СОХРАНЯЕТСЯ, kvRestored=0, фантомного SKEW нет (гибрид SSM/Mamba)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.900 (7200/8000) > 0.100' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 6000 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 8000' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 6000 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 6000 | prompt eval time =  10000.00 ms / 8000 tokens (    1.25 ms per token,   800.00 tokens per second)' },
    { ts: T(4), msg: 'slot print_timing: id  0 | task 6000 | n_gen =    101, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(5), msg: 'slot print_timing: id  0 | task 6000 |        eval time =   99864.69 ms /  2574 tokens (   38.81 ms per token,    25.76 tokens per second)' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 7200, 'LCP-совпадение сохранено (факт, а не источник KV)');
  assert.strictEqual(s.task.lcpRatio, 0.9, 'lcpRatio сохранён');
  assert.strictEqual(s.task.kvRestoredTokens, 0, 'ничего не восстановлено');
  assert.strictEqual(s.task.slotSource, 'hot-vram', 'slotSource сохранён (не обнулён в empty)');
  assert.strictEqual(s.task.reprocessReason, 'FULL_REPROCESS', 'причина: полный пересчёт');
  assert.strictEqual(s.task.deltaTokens, 8000, 'пересчитан весь промпт');
  const h = s.history[0];
  assert.ok(h, 'таск в истории');
  assert.strictEqual(h.lcp, 7200, 'lcp в истории сохранён');
  assert.strictEqual(h.kvReuse, 0, 'kvReuse в истории = 0');
  assert.strictEqual(h.cpSkew, 0, 'нет фантомного SKEW (пересчитан весь промпт, а не хвост)');
  assert.strictEqual(h.reprocessReason, 'FULL_REPROCESS', 'reprocessReason в истории');
});

test('L7b: forcing full re-processing БЕЗ LCP-совпадения — cold start (KV MISS)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 6100 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 8000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 6100 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 6100 | prompt eval time =  10000.00 ms / 8000 tokens (    1.25 ms per token,   800.00 tokens per second)' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 6100 | n_gen =    101, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(4), msg: 'slot print_timing: id  0 | task 6100 |        eval time =   99864.69 ms /  2574 tokens (   38.81 ms per token,    25.76 tokens per second)' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 0, 'LCP не было');
  assert.strictEqual(s.task.kvRestoredTokens, 0, 'ничего не восстановлено');
  assert.strictEqual(s.task.slotSource, 'empty', 'cold start');
  assert.strictEqual(s.task.reprocessReason, 'FULL_REPROCESS', 'причина: полный пересчёт');
  assert.strictEqual(s.task.deltaTokens, 8000, 'пересчитан весь промпт');
  const h = s.history[0];
  assert.ok(h, 'таск в истории');
  assert.strictEqual(h.cpSkew, 0, 'нет SKEW');
});

test('L8: Δ LOST — только при reuse > 0: отдельный запрос → «—», реальный трим считается', () => {
  // histCells — чистая клиентская функция: извлекаем из исходника и гоняем напрямую
  const src = fs.readFileSync(WWW, 'utf8');
  const m = src.match(/function histCells[\s\S]*?\n    }/);
  assert.ok(m, 'histCells found in www.js');
  const num = (v, def = 0) => { const n = Number(v); return Number.isFinite(n) ? n : def; };
  const esc = v => String(v ?? '');
  const t = (k) => k; // i18n-заглушка: title-атрибуты не проверяются
  const histCells = new Function('num', 'esc', 't', m[0] + '\nreturn histCells;')(num, esc, t);

  // Кейс из скриншота: prev ход input=100 + decode=4 (expected=104), текущий —
  // отдельный короткий запрос без совпадения префикса (reuse=0, cold)
  const r = histCells(
    { input: 50, kvReuse: 0, lcp: 0, slotSource: 'empty', checkpointRestore: false, cpSkew: 0, divergeInfo: null },
    { input: 100, decode: 4 }
  );
  assert.ok(r.lostHtml.includes('—'), 'тире вместо ложной -104');
  assert.ok(!r.lostHtml.includes('badge-lost-warn'), 'нет warn-бейджа');

  // Реальная обрезка контекста: продолжение, reuse=4000 < expected=5200
  const r2 = histCells(
    { input: 4100, kvReuse: 4000, lcp: 4000, slotSource: 'hot-vram', checkpointRestore: false, cpSkew: 0, divergeInfo: null },
    { input: 5000, decode: 200 }
  );
  assert.ok(r2.lostHtml.includes('badge-lost-warn'), 'реальная потеря 5200−4000 посчитана');

  // Полное совпадение: reuse == expected → «✓ 0»
  const r3 = histCells(
    { input: 110, kvReuse: 104, lcp: 104, slotSource: 'hot-vram', checkpointRestore: false, cpSkew: 0, divergeInfo: null },
    { input: 100, decode: 4 }
  );
  assert.ok(r3.lostHtml.includes('✓ 0'), 'полное совпадение — ✓ 0');
});

test('PLAYBACK: load-log в LIVE, stdin-строка в PLAYBACK только на диск, live возвращает LIVE', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-pb-'));
  const logFile = path.join(dir, 'llama.log');
  const upload = path.join(dir, 'upload.log');
  const port = 30000 + Math.floor(Math.random() * 20000);
  fs.writeFileSync(upload, [
    `[${T(0)}] srv update: - cache state: 1 prompts, 1.000 MiB (limits: 40960.000 MiB, 123136 tokens, 42949672960 est)`,
    `[${T(1)}] slot create_check: id  0 | task 1 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)`
  ].join('\n') + '\n');
  // LIVE-процесс: stdin-пайп, мёртвый URL сервера (pollServer не мешает)
  const child = spawn('node', [WWW, '--url', 'http://127.0.0.1:9'], {
    env: { ...process.env, LLAMA_LOG_FILE: logFile, WEB_PORT: String(port) },
    stdio: ['pipe', 'pipe', 'pipe']
  });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const getState = async () => (await fetch(`http://127.0.0.1:${port}/api/state`)).json();
  try {
    let s = null;
    for (let i = 0; i < 50 && !s; i++) {
      try { const r = await fetch(`http://127.0.0.1:${port}/api/state`); if (r.ok) s = await r.json(); } catch (_) {}
      if (!s) await sleep(200);
    }
    assert.ok(s, 'сервер не запустился');
    assert.strictEqual(s.mode, 'LIVE');

    // Загрузка лога в LIVE -> PLAYBACK
    const lr = await fetch(`http://127.0.0.1:${port}/api/load-log`, { method: 'POST', body: fs.readFileSync(upload, 'utf8') });
    assert.strictEqual(lr.status, 200);
    for (let i = 0; i < 50; i++) {
      s = await getState();
      if (s.player && s.player.finished) break;
      await sleep(200);
    }
    assert.strictEqual(s.mode, 'PLAYBACK');
    assert.ok(s.events.length > 0, 'воспроизведение дало события');
    const evBefore = s.events.length, histBefore = s.history.length;

    // stdin-строка во время PLAYBACK: на диск, state не меняется
    child.stdin.write('slot create_check: id  7 | task 9 | created context checkpoint 1 of 32 (pos_min = 1, pos_max = 1, n_tokens = 2, size = 0.100 MiB)\n');
    await sleep(600);
    s = await getState();
    assert.strictEqual(s.mode, 'PLAYBACK');
    assert.strictEqual(s.events.length, evBefore, 'PLAYBACK не парсит живые строки');
    assert.strictEqual(s.history.length, histBefore);
    assert.ok(fs.readFileSync(logFile, 'utf8').includes('task 9'), 'живая строка должна попасть в LOG_FILE');

    // live -> сброс, режим LIVE
    const ar = await fetch(`http://127.0.0.1:${port}/api/player`, { method: 'POST', body: JSON.stringify({ action: 'live' }) });
    assert.strictEqual(ar.status, 200);
    await sleep(300);
    s = await getState();
    assert.strictEqual(s.mode, 'LIVE');
    assert.strictEqual(s.events.length, 0, 'состояние сброшено');
    assert.strictEqual(s.history.length, 0);

    // Новая stdin-строка в LIVE обрабатывается
    child.stdin.write('slot create_check: id  0 | task 5 | created context checkpoint 1 of 32 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)\n');
    await sleep(600);
    s = await getState();
    assert.strictEqual(s.mode, 'LIVE');
    assert.ok(s.events.length > 0, 'LIVE-строка должна быть обработана');
  } finally {
    child.kill();
  }
});

test('L9: POST только из своего origin; liveCapable; тулбар плеера в HTML_PAGE (#4)', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-l9-'));
  const logFile = path.join(dir, 'llama.log');
  const port = 30000 + Math.floor(Math.random() * 20000);
  const child = spawn('node', [WWW, '--url', 'http://127.0.0.1:9'], {
    env: { ...process.env, LLAMA_LOG_FILE: logFile, WEB_PORT: String(port) },
    stdio: ['pipe', 'pipe', 'pipe']
  });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  try {
    let s = null;
    for (let i = 0; i < 50 && !s; i++) {
      try { const r = await fetch(`http://127.0.0.1:${port}/api/state`); if (r.ok) s = await r.json(); } catch (_) {}
      if (!s) await sleep(200);
    }
    assert.ok(s, 'сервер не запустился');
    assert.strictEqual(s.liveCapable, true, 'LIVE: stdin-поток есть');

    // Страница: тулбар плеера теперь в HTML_PAGE (#4)
    const page = await (await fetch(`http://127.0.0.1:${port}/`)).text();
    assert.ok(!page.includes('class="ctl-bar"'), 'нет старого .ctl-bar в HTML_PAGE');
    assert.ok(page.includes('class="player-toolbar"'), 'тулбар плеера в HTML_PAGE (#4)');
    assert.ok(page.includes('id="ctl-play"'), 'кнопка play/pause в HTML_PAGE');

    // Чужой origin — 403: simple POST (text/plain, без preflight) всё равно
    // выполнил бы сброс, даже если браузер заблокировал чтение ответа
    const evil = await fetch(`http://127.0.0.1:${port}/api/clear`, { method: 'POST', body: '', headers: { Origin: 'http://evil.example' } });
    assert.strictEqual(evil.status, 403, 'чужой origin отклонён');
    // Свой origin — 200
    const own = await fetch(`http://127.0.0.1:${port}/api/clear`, { method: 'POST', body: '', headers: { Origin: `http://127.0.0.1:${port}` } });
    assert.strictEqual(own.status, 200, 'свой origin пропущен');
    // Без Origin (curl/CLI) — 200
    const noOrigin = await fetch(`http://127.0.0.1:${port}/api/clear`, { method: 'POST', body: '' });
    assert.strictEqual(noOrigin.status, 200, 'без Origin (CLI) пропущен');
  } finally {
    child.kill();
  }
});

test('L10: idle-спам не пишется в LOG_FILE, обычные строки — да', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-l10-'));
  const logFile = path.join(dir, 'llama.log');
  const port = 30000 + Math.floor(Math.random() * 20000);
  const child = spawn('node', [WWW, '--url', 'http://127.0.0.1:9'], {
    env: { ...process.env, LLAMA_LOG_FILE: logFile, WEB_PORT: String(port) },
    stdio: ['pipe', 'pipe', 'pipe']
  });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  try {
    for (let i = 0; i < 50; i++) {
      try { if ((await fetch(`http://127.0.0.1:${port}/api/state`)).ok) break; } catch (_) {}
      await sleep(200);
    }
    child.stdin.write('srv  update: update_slots: all slots are idle\n');
    child.stdin.write('slot create_check: id  0 | task 3 | created context checkpoint 1 of 32 (pos_min = 5, pos_max = 5, n_tokens = 6, size = 0.100 MiB)\n');
    await sleep(800);
    const content = fs.readFileSync(logFile, 'utf8');
    assert.ok(content.includes('task 3'), 'обычная строка записана');
    assert.ok(!content.includes('all slots are idle'), 'idle-спам не записан');
  } finally {
    child.kill();
  }
});

test('L11: replay — liveCapable=false (нет stdin-потока)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' }
  ]);
  assert.strictEqual(s.liveCapable, false, 'в replay «⏭ В LIVE» невозможен');
});

// ===== N: IDLE-мердж (кольцо + легенда → зелёный KV) и анти-дребезг декода =====

test('N1: IDLE/TRUNC-мердж — логика в renderState (кольцо и легенда синхронно)', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes("const idleMerge = stage === 'IDLE'"), 'флаг idleMerge');
  assert.ok(src.includes('const isTrunc = !!t.isTruncated && finalVal > 0'), 'флаг isTrunc (сдвиг контекста)');
  assert.ok(src.includes('const merge = idleMerge || isTrunc'), 'merge = IDLE || TRUNC');
  assert.ok(src.includes('const liveNpt = (s.mode === \'LIVE\' && s.is_processing) ? num(s.n_prompt_tokens, 0) : 0'), 'LIVE: n_prompt_tokens из /slots');
  assert.ok(src.includes('const liveShift = liveNpt > 0 && liveNpt < anatomySum - Math.max(64, anatomySum * 0.02)'), 'liveShift: сдвиг контекста по /slots (толеранс 2%/64)');
  assert.ok(src.includes('const merge = idleMerge || isTrunc || liveShift'), 'merge = IDLE || TRUNC || LIVE-SHIFT');
  assert.ok(src.includes('const rawUsed = liveNpt > 0 ? liveNpt : (isTrunc ? finalVal : anatomySum)'), 'база: LIVE → n_prompt_tokens, TRUNC → finalSlotTokens, иначе анатомия');
  assert.ok(src.includes('merge ? rawUsed : cachedVal'), 'кольцо: в IDLE/TRUNC весь объём в KV');
  assert.ok(src.includes('(merge ? 0 : deltaVal)'), 'легенда: в IDLE/TRUNC PREFILL = 0');
  assert.ok(src.includes('(merge ? 0 : genVal)'), 'легенда: в IDLE/TRUNC DECODE = 0');
});

test('N2: IDLE-мердж — предпосылка: после DONE анатомия запроса сохранена в снапшоте', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 700 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 5000' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 700 | prompt eval time =  1000.00 ms / 5000 tokens (    0.20 ms per token,  5000.00 tokens per second)' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 700 | n_gen =    101, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(3), msg: 'slot print_timing: id  0 | task 700 |        eval time =   99864.69 ms /  2574 tokens (   38.81 ms per token,    25.76 tokens per second)' }
  ]);
  // Снапшот пишется через 2с — после таймера DONE→IDLE (1.5с): фаза IDLE,
  // а анатомия запроса сохранена в task — ровно то, что мердж потребит
  assert.strictEqual(s.currentStage, 'IDLE', 'после DONE фаза перешла в IDLE');
  assert.ok(s.task.deltaTokens > 0, 'prefill-анатомия сохранена (мердж её потребит)');
  assert.strictEqual(s.task.generatedTokens, 2574, 'decode-анатомия сохранена (мердж её потребит)');
});

test('N3: строка декода не откатывает generatedTokens (монотонность против /slots)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 701 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 1000' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 701 | n_gen =    150, tg =  22.70 t/s, tg_3s =  22.92 t/s' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 701 | n_gen =    120, tg =  22.70 t/s, tg_3s =  22.92 t/s' }
  ]);
  assert.strictEqual(s.task.generatedTokens, 150, 'отстающая строка лога не откатывает счётчик');
});

test('N4: метка чекпоинта — фиксированный радиус (без радиального «попа» +6px)', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(!src.includes('R2 + born * 6'), 'радиальный поп +6px удалён');
});

test('N5: MIN_ARC_PX — инвариант длины дуги для кольца VRAM и R3', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('const MIN_ARC_PX = 2'), 'константа инварианта объявлена');
  assert.ok(src.includes('cachedAngle * R >= MIN_ARC_PX'), 'зелёный (cached) сегмент гвардится');
  assert.ok(src.includes('deltaAngle * R >= MIN_ARC_PX'), 'жёлтый (delta) сегмент гвардится');
  assert.ok(src.includes('genAngle * R >= MIN_ARC_PX'), 'фиолетовый (gen) сегмент гвардится');
  assert.ok(src.includes('seg.len * R3 * Math.PI * 2 < MIN_ARC_PX'), 'кольцо R3 гвардится тем же инвариантом');
  assert.ok(src.includes('if (seg.removedAt || seg.len * R3 * Math.PI * 2 < MIN_ARC_PX) continue;'), 'hover: невидимые (sub-pixel) сегменты не дают tooltip — draw и hit-test совпадают');
});

test('N6: умирающая метка чекпоинта — без радиального drift (+6px при death)', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(!src.includes('R2 + death * 6'), 'drift радиуса при death удалён (затухание только через alpha)');
});

test('N7: decode — чистая дуга, без точки/шарика (субпиксельные дуги не рисуются)', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('if (anim.gen > 2 && genAngle * R >= MIN_ARC_PX)'), 'decode — только дуга при длине >= MIN_ARC_PX');
  assert.ok(!src.includes("ctx.fillStyle = '#a855f7'"), 'точка/шарик декода удалён (нет fill)');
  assert.ok(!src.includes('const midA = sA + genAngle * 0.5'), 'расчёт середины дуги для точки удалён');
});

test('N8: золотая раскладка — VRAM (R=201,LW=26), чекпоинты (R2=167,LW2=16), RAM (R3=146,LW3=10); вихрь-луч', () => {
  const src = fs.readFileSync(WWW, 'utf8');
  assert.ok(src.includes('const R = 201, LW = 26'), 'внешнее кольцо VRAM: R=201, LW=26');
  assert.ok(src.includes('const R2 = 167, LW2 = 16'), 'среднее кольцо чекпоинтов: R2=167, LW2=16');
  assert.ok(src.includes('const R3 = 146, LW3 = 10'), 'внутреннее кольцо Host RAM: R3=146, LW3=10');
  assert.ok(src.includes('const BEAM_SWEEP = 0.16'), 'вихрь: константа тангенциального сдвига луча');
  assert.ok(src.includes('const aOut = a + BEAM_SWEEP'), 'конец луча смещён по часовой стрелке');
  assert.ok(src.includes('ctx.moveTo(CX + Math.cos(a) * rIn'), 'beam: старт от кольца чекпоинтов (Host RAM)');
  assert.ok(src.includes('ctx.lineTo(CX + Math.cos(aOut) * rOut'), 'beam: финиш во внешнем кольце VRAM (тангенциально)');
  assert.ok(src.includes('until = Math.max(until, paintRing3(model.ram, now));'), 'порядок отрисовки: R3 первым, VRAM поверх');
  assert.ok(src.includes('dist >= R2 - LW2 / 2 - 3 && dist <= R2 + LW2 / 2 + 3'), 'hover R2: диапазон с допуском ±3px');
  assert.ok(src.includes('dist >= R3 - LW3 / 2 - 3 && dist <= R3 + LW3 / 2 + 3'), 'hover R3: диапазон с допуском ±3px');
});

// ===== T: третий VERDICT (c25030496) — точные источники /slots и чекпоинты =====

// T1 (fix 1): f_sim_best без (N/M) — точный count берётся из "checking sim"
// того же кандидата (обе строки печатают один и тот же float f_sim_best).
test('T1: sim_best — точный LCP из "checking sim" того же кандидата (равенство ratio)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.999 (47320/47375) > 0.100' },
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.999 (> 0.100 thold), f_keep = 1.000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 4660 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 47375' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 47320, 'точный count из checking sim, не оценка');
  assert.strictEqual(s.task.lcpRatio, 47320 / 47375, 'ratio пересчитан против нового промпта');
  assert.strictEqual(s.task.slotSource, 'hot-vram');
});

// T2 (fix 1 fallback): строки "checking sim" не было (trace-уровень не дошёл)
// → tokens = 0 → оценка round(prompt * ratio) в initNewTask.
test('T2: sim_best без "checking sim" — оценка round(prompt*ratio)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.875 (> 0.100 thold), f_keep = 1.000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 4661 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 80000' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 70000, 'оценка round(80000*0.875)');
  assert.strictEqual(s.task.lcpRatio, 0.875);
});

// T3 (fix 2): у RAM-кандидата ratio = f_sim (lcp / new_prompt_size), а НЕ
// lcp / длина кандидата (это f_keep — другая величина).
test('T3: RAM-кандидат — ratio = f_sim, lcpRatio против НОВОГО промпта', () => {
  const s = replay([
    { ts: T(0), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.001 (10/47375) > 0.100' },
    { ts: T(1), msg: 'srv          load:    - prompt with length    47328, lcp =    47320, f_keep = 1.000, f_sim = 0.999' },
    { ts: T(1), msg: 'srv          load:  - found better prompt with f_keep = 1.000, f_sim = 0.999' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 4662 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 47375' }
  ]);
  assert.strictEqual(s.task.lcpTokens, 47320, 'точный lcp кандидата');
  assert.strictEqual(s.task.lcpRatio, 47320 / 47375, 'ratio против нового промпта, не 47320/47328');
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram');
});

// T4 (fix 3): "slot context shift" + truncated=1 — это SHIFTED, не OVERFLOW.
// Upstream порядок: print_timings (eval time) ПЕРЕД release (stop processing) —
// строка истории пишется в eval-time ветке, флаги патчатся в stop-processing.
test('T4: context shift + truncated=1 — SHIFTED, не OVERFLOW (патч истории)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7030 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 123900' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7030 | slot context shift, n_keep = 0, n_left = 100, n_discard = 50' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7030 | eval time = 1000.000 ms / 100 tokens, with 100.000 tokens per second' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 7030 | stop processing: n_tokens = 123999, truncated = 1' }
  ]);
  assert.strictEqual(s.task.ctxShifted, true);
  assert.strictEqual(s.task.isTruncated, true);
  const h = s.history[0];
  assert.strictEqual(h.isTruncated, true, 'патч: stop processing приходит после eval time');
  assert.strictEqual(h.ctxShifted, true, 'SHIFTED, не OVERFLOW');
  assert.ok(s.events.find(e => e.tag === 'CONTEXT SHIFT'), 'эвент CONTEXT SHIFT');
});

// T5 (fix 3): truncated=1 БЕЗ shift — OVERFLOW (переполнение).
test('T5: truncated=1 без shift — OVERFLOW', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 7031 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 123900' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 7031 | eval time = 1000.000 ms / 100 tokens, with 100.000 tokens per second' },
    { ts: T(2), msg: 'slot operator(): id  0 | task 7031 | stop processing: n_tokens = 124000, truncated = 1' }
  ]);
  assert.strictEqual(s.task.ctxShifted, false);
  assert.strictEqual(s.task.isTruncated, true);
  const h = s.history[0];
  assert.strictEqual(h.isTruncated, true);
  assert.strictEqual(h.ctxShifted, false, 'OVERFLOW, не SHIFTED');
  assert.ok(s.events.find(e => e.tag === 'OVERFLOW'), 'эвент OVERFLOW');
});

// T6 (fix 6a): "superseding context checkpoint at n_tokens = N" — новый
// чекпоинт на той же точке заменяет старый: метка стойки снята (superseded).
test('T6: superseding — чекпоинт на той же точке снят (removeReason=superseded)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 64 (pos_min = 39999, pos_max = 39999, n_tokens = 40000, size = 150.000 MiB)' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | superseding context checkpoint at n_tokens = 40000' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 64 (pos_min = 40000, pos_max = 40000, n_tokens = 40001, size = 150.000 MiB)' }
  ]);
  const old = s.ramCache.slots['0:39999:39999'];
  assert.ok(old, 'старый чекпоинт ещё на стойке (окно 2с)');
  assert.strictEqual(old.removeReason, 'superseded');
  assert.strictEqual(old.removedAt, Date.parse(T(1)));
  assert.ok(s.ramCache.slots['0:40000:40000'], 'новый чекпоинт создан');
  assert.strictEqual(s.ramCache.cpCurrent, 2, 'N=2 авторитетен (resync)');
  assert.ok(s.events.find(e => e.tag === 'KV SUPERSEDE'), 'эвент KV SUPERSEDE');
});

// T7 (fix 6b): "found better prompt" — prompt = std::move(it_best->prompt):
// список чекпоинтов переехал вместе с промптом, метки прежнего промпта
// слота устарели → сняты (prompt-replace), LCP таска = lcp кандидата.
test('T7: found better prompt — чекпоинты слота сняты (removeReason=prompt-replace)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 64 (pos_min = 100, pos_max = 100, n_tokens = 101, size = 1.500 MiB)' },
    { ts: T(1), msg: 'slot get_availabl: id  0 | task -1 |  - checking sim = 0.001 (10/20000) > 0.100' },
    { ts: T(2), msg: 'srv          load:    - prompt with length    15000, lcp =    14000, f_keep = 0.933, f_sim = 0.700' },
    { ts: T(2), msg: 'srv          load:  - found better prompt with f_keep = 0.933, f_sim = 0.700' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 4670 | new prompt, n_ctx_slot = 128000, n_keep = 0, task.n_tokens = 20000' }
  ]);
  const old = s.ramCache.slots['0:100:100'];
  assert.ok(old, 'метка прежнего промпта ещё на стойке (окно 2с)');
  assert.strictEqual(old.removeReason, 'prompt-replace');
  assert.strictEqual(old.removedAt, Date.parse(T(2)));
  assert.strictEqual(s.ramCache.cpCurrent, 0, 'живых чекпоинтов нет');
  assert.strictEqual(s.task.lcpTokens, 14000, 'LCP кандидата');
  assert.strictEqual(s.task.slotSource, 'prompt-cache-ram');
});

// T8 (п.5): CHECKPOINT LOOP watchdog — 5 одинаковых erase-строк подряд
// (тот же task, слот, pos_min/pos_max, тип) — сигнатура upstream
// checkpoint livelock. 4 строки — порог не достигнут, эвента нет.
test('T8: CHECKPOINT LOOP watchdog — 5 одинаковых erase -> эвент, 4 -> нет', () => {
  const erase = 'slot operator(): id  0 | task 327 | erased invalidated context checkpoint (pos_min = 100, pos_max = 100, n_tokens = 101, n_swa = 0, pos_next = 0, size = 1.500 MiB)';
  const mk = (n) => Array.from({ length: n }, (_, i) => ({ ts: T(i), msg: erase }));
  const s5 = replay(mk(5));
  const ev = s5.events.find(e => e.tag === 'CHECKPOINT LOOP');
  assert.ok(ev, 'эвент CHECKPOINT LOOP на 5-м erase');
  assert.strictEqual(ev.taskId, '327', 'привязан к таску строки');
  const s4 = replay(mk(4));
  assert.ok(!s4.events.find(e => e.tag === 'CHECKPOINT LOOP'), '4 erase — эвента нет');
});

// T9 (п.6): CLI --play — режим обязан быть 'PLAYBACK' (фронт знает только
// LIVE/PLAYBACK/REPLAY; отдельный 'PLAY' показывал зелёный ONLINE).
test('T9: --play -> state.mode === PLAYBACK', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-play-'));
  const log = path.join(dir, 'log.txt');
  fs.writeFileSync(log,
    `[${T(0)}] slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100\n` +
    `[${T(1)}] slot operator(): id  0 | task 327 | new prompt processing, n_tokens = 100, memory = 100.000 MiB\n`);
  const port = 40000 + Math.floor(Math.random() * 9000);
  let stderr = '';
  const child = spawn('node', [WWW, '--play', log, '--speed', '100', '--port', String(port)], {
    stdio: ['ignore', 'ignore', 'pipe'],
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' }
  });
  child.stderr.on('data', d => { stderr += d; if (stderr.length > 8000) stderr = stderr.slice(-8000); });
  try {
    let up = false;
    for (let i = 0; i < 100 && !up; i++) {
      await new Promise(r => setTimeout(r, 100));
      try { up = (await fetch(`http://127.0.0.1:${port}/api/state`)).ok; } catch (_) { /* ещё стартует */ }
    }
    assert.ok(up, 'сервер поднялся (stderr: ' + stderr + ')');
    const s = await (await fetch(`http://127.0.0.1:${port}/api/state`)).json();
    assert.strictEqual(s.mode, 'PLAYBACK');
  } finally {
    child.kill('SIGKILL');
  }
});

// T10 (freeze P0): supersede — РОВНО ОДИН чекпоинт снимается (старейший по
// createdAt), второй с тем же n_tokens жив. Строка upstream не несёт
// pos_min/pos_max — identity по slot + n_tokens, недоопределение безопаснее
// лишнего удаления.
test('T10: supersede removes exactly one checkpoint (oldest createdAt)', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 40000' },
    { ts: T(1), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 1 of 64 (pos_min = 100, pos_max = 100, n_tokens = 40000, size = 1.500 MiB)' },
    { ts: T(2), msg: 'slot create_check: id  0 | task 327 | created context checkpoint 2 of 64 (pos_min = 200, pos_max = 200, n_tokens = 40000, size = 2.000 MiB)' },
    { ts: T(3), msg: 'slot create_check: id  0 | task 327 | superseding context checkpoint at n_tokens = 40000' }
  ]);
  assert.ok(s.ramCache.slots['0:100:100'].removedAt, 'старейший снят');
  assert.strictEqual(s.ramCache.slots['0:100:100'].removeReason, 'superseded');
  assert.strictEqual(s.ramCache.slots['0:200:200'].removedAt, undefined, 'второй жив');
  assert.strictEqual(s.ramCache.cpCurrent, 1, 'ровно один остался');
});

// T11 (freeze P0): prompt_save — служебная строка на границе тасков:
// currentStage не двигается (DECODE остаётся DECODE), событие НЕ в ленте
// сразу — буферизуется (pendingSysEvents) и приклеивается к НОВОМУ таску
// при следующем "new prompt" (#7).
test('T11: prompt_save does not move currentStage and is buffered', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | n_decoded = 50, tg = 45.6 t/s' },
    { ts: T(2), msg: 'prompt_save: saving prompt with length 12345, total state size = 123.456 MiB' }
  ]);
  assert.strictEqual(s.currentStage, 'DECODE', 'prompt_save не двигает stage');
  const ev = s.events.find(e => e.stage === 'PREFIX' && e.tag === 'RAM CACHE');
  assert.strictEqual(ev, undefined, 'событие буферизуется, в ленте его нет');
});

// T11b (#7): prompt_save → новый промпт — буферизованное событие
// приклеивается к НОВОМУ таску (taskId нового), а не к предыдущему.
test('T11b: prompt_save attaches to the NEW task on next prompt', () => {
  const s = replay([
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 1000' },
    { ts: T(1), msg: 'slot operator(): id  0 | task 327 | n_decoded = 50, tg = 45.6 t/s' },
    { ts: T(2), msg: 'prompt_save: saving prompt with length 12345, total state size = 123.456 MiB' },
    { ts: T(3), msg: 'slot operator(): id  0 | task 328 | new prompt, n_ctx_slot = 124160, n_keep = 0, task.n_tokens = 2000' }
  ]);
  const ev = s.events.find(e => e.stage === 'PREFIX' && e.tag === 'RAM CACHE');
  assert.ok(ev, 'событие Снимок диалога в ленте после нового промпта');
  assert.strictEqual(ev.taskId, '328', 'событие приклеено к НОВОМУ таску');
  const onOld = s.events.find(e => e.stage === 'PREFIX' && e.tag === 'RAM CACHE' && e.taskId === '327');
  assert.strictEqual(onOld, undefined, 'не приклеено к предыдущему таску');
});

// ===== Финальный чек-лист: скорость, MONITOR_HOST, 413 =====

// #6: скорость переживает load-log (4x → load-log → 4x)
test('#6: скорость переживает load-log (4x → load-log → 4x)', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-speed-'));
  const mkLog = (name, n) => {
    const p = path.join(dir, name);
    const lines = [`[${T(0)}] srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'`];
    for (let i = 1; i <= n; i++) lines.push(`[${T(i)}] slot operator(): id  0 | task ${i} | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100`);
    fs.writeFileSync(p, lines.join('\n') + '\n');
    return p;
  };
  const port = 30000 + Math.floor(Math.random() * 20000);
  const child = spawn('node', [WWW, '--play', mkLog('a.log', 3), '--speed', '4', '--port', String(port)], {
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' },
    stdio: ['pipe', 'pipe', 'pipe']
  });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  try {
    let s = null;
    for (let i = 0; i < 50 && !s; i++) {
      try { const r = await fetch(`http://127.0.0.1:${port}/api/state`); if (r.ok) s = await r.json(); } catch (_) {}
      if (!s) await sleep(200);
    }
    assert.ok(s, 'сервер не запустился');
    assert.strictEqual(s.player.speed, 4, 'старт --speed 4');
    const lr = await fetch(`http://127.0.0.1:${port}/api/load-log`, { method: 'POST', body: fs.readFileSync(mkLog('b.log', 3), 'utf8') });
    assert.strictEqual(lr.status, 200);
    for (let i = 0; i < 50; i++) {
      s = await (await fetch(`http://127.0.0.1:${port}/api/state`)).json();
      if (s.player && s.player.finished) break;
      await sleep(200);
    }
    assert.strictEqual(s.player.speed, 4, 'скорость пережила load-log');
  } finally { child.kill('SIGKILL'); }
});

// #15: MONITOR_HOST по умолчанию 127.0.0.1 (не 0.0.0.0)
test('#15: MONITOR_HOST по умолчанию 127.0.0.1 (не 0.0.0.0)', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-host-'));
  const log = path.join(dir, 'llama.log');
  fs.writeFileSync(log, `[${T(0)}] srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'\n`);
  const port = 30000 + Math.floor(Math.random() * 20000);
  const env = { ...process.env, LLAMA_LOG_FILE: '/dev/null' };
  delete env.MONITOR_HOST;
  const child = spawn('node', [WWW, '--play', log, '--port', String(port)], { env, stdio: ['pipe', 'pipe', 'pipe'] });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  try {
    let up = false;
    for (let i = 0; i < 50 && !up; i++) {
      try { up = (await fetch(`http://127.0.0.1:${port}/api/state`)).ok; } catch (_) {}
      if (!up) await sleep(200);
    }
    assert.ok(up, 'сервер не запустился');
    // /proc/net/tcp: local_address = hexIP:hexPORT; 127.0.0.1 = 0100007F
    const hexPort = port.toString(16).toUpperCase().padStart(4, '0');
    const tcp = fs.readFileSync('/proc/net/tcp', 'utf8');
    const line = tcp.split('\n').find(l => l.includes(':' + hexPort + ' '));
    assert.ok(line, 'порт в /proc/net/tcp');
    assert.ok(line.includes('0100007F:' + hexPort), 'бинд на 127.0.0.1, а не 0.0.0.0');
  } finally { child.kill('SIGKILL'); }
});

// #10: load-log > 2 МБ — контролируемый 413 (не OOM)
test('#10: load-log > 2 МБ — контролируемый 413 (не OOM)', async () => {
  const port = 30000 + Math.floor(Math.random() * 20000);
  const child = spawn('node', [WWW, '--url', 'http://127.0.0.1:9'], {
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null', WEB_PORT: String(port) },
    stdio: ['pipe', 'pipe', 'pipe']
  });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  try {
    let up = false;
    for (let i = 0; i < 50 && !up; i++) {
      try { up = (await fetch(`http://127.0.0.1:${port}/api/state`)).ok; } catch (_) {}
      if (!up) await sleep(200);
    }
    assert.ok(up, 'сервер не запустился');
    const r = await fetch(`http://127.0.0.1:${port}/api/load-log`, { method: 'POST', body: 'x'.repeat(2 * 1024 * 1024 + 1) });
    assert.strictEqual(r.status, 413, 'контролируемый 413');
  } finally { child.kill('SIGKILL'); }
});

// ═════════════════════════════════════════════════════════════════════
// ФИНАЛЬНЫЙ PASS: MTP dynamic
//   - specConfig.draftNMax — из startup-лога ("- n_max=N"), без зашитых чисел
//   - MTP-бары — ТОЛЬКО из state.specMtp.posAcc (не state.task.specPosAcc)
//   - UI-render harness: реальный клиентский скрипт страницы в Node (DOM-shim)
// ═════════════════════════════════════════════════════════════════════

const MTP_STARTUP = (nMax, t0 = 0) => [
  { ts: T(t0), msg: "spec common_specu: adding speculative implementation 'draft-mtp'" },
  { ts: T(t0 + 1), msg: `spec common_specu: - n_max=${nMax}, n_min=0, p_min=0.80, n_embd=5120, backend_sampling=1` }
];
// Реальный формат строки (demo.log, statistics draft-mtp)
const MTP_STATS = { ts: T(2), msg: 'spec common_specu: statistics        draft-mtp: #calls(b,g,a) =   60  16349  11027, #gen drafts =  11027, #acc drafts = 10748, #gen tokens =  34819, #acc tokens = 32592, #mean acc len = 3.96, #acc rate/pos = (0.975, 0.698, 0.531, 0.415, 0.337), dur(b,g,a) = 0.057, 148600.577, 10.084 ms' };

test('MTP state: startup "- n_max=5" → specConfig.draftNMax = 5', () => {
  assert.strictEqual(replay(MTP_STARTUP(5)).specConfig.draftNMax, 5);
});

test('MTP state: synthetic startup "- n_max=8" → specConfig.draftNMax = 8 (без хардкода)', () => {
  assert.strictEqual(replay(MTP_STARTUP(8)).specConfig.draftNMax, 8);
});

test('MTP state: ngram-mod startup-строки до draft-mtp не ломают парсинг n_max', () => {
  const s = replay([
    { ts: T(0), msg: "spec common_specu: adding speculative implementation 'ngram-mod'" },
    { ts: T(1), msg: 'spec common_specu: - n_match=24, n_max=64, n_min=48' },
    ...MTP_STARTUP(5, 2)
  ]);
  assert.strictEqual(s.specConfig.draftNMax, 5);
});

test('MTP state: лог без startup-строк (demo.log) → draftNMax = null, posAcc из statistics', () => {
  const s = replay([MTP_STATS]);
  assert.strictEqual(s.specConfig.draftNMax, null);
  assert.deepStrictEqual(s.specMtp.posAcc, ['0.975', '0.698', '0.531', '0.415', '0.337']);
});

test('MTP state: statistics draft-mtp → сессионные счётчики specMtp', () => {
  const s = replay([...MTP_STARTUP(5), MTP_STATS]);
  assert.strictEqual(s.specConfig.draftNMax, 5);
  assert.strictEqual(s.specMtp.accTokens, 32592);
  assert.strictEqual(s.specMtp.genTokens, 34819);
  assert.strictEqual(s.specMtp.meanLen, 3.96);
  assert.deepStrictEqual(s.specMtp.calls, [60, 16349, 11027]);
});

// ─── UI-render harness (обязательный тест 5) ───
// Клиентский скрипт берётся из РЕАЛЬНОЙ страницы (сервер --play, GET /) —
// тот же код, что работает в браузере. DOM — shim, renderState вызывается
// напрямую: именно этот слой содержал баг SPEC TOTAL → Math.min(12, ...).
// Один и тот же скрипт для всех кейсов — код между тестами не меняется.

async function loadClientScript() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-ui-'));
  const log = path.join(dir, 'log.txt');
  fs.writeFileSync(log, `[${T(0)}] srv load_model: initializing, n_slots = 1, n_ctx_slot = 123136, kv_unified = 'false'\n`);
  const port = 30000 + Math.floor(Math.random() * 20000);
  const child = spawn('node', [WWW, '--play', log, '--port', String(port)], {
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' },
    stdio: ['pipe', 'pipe', 'pipe']
  });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  try {
    let html = null;
    for (let i = 0; i < 50 && !html; i++) {
      try { const r = await fetch(`http://127.0.0.1:${port}/`); if (r.ok) html = await r.text(); } catch (_) {}
      if (!html) await sleep(200);
    }
    assert.ok(html, 'страница не отдаётся');
    const m = html.match(/<script>([\s\S]*?function renderState[\s\S]*?)<\/script>/);
    assert.ok(m, 'главный <script> (renderState) не найден');
    // Экспорт renderState из IIFE (инъекция в хвост — код страницы не меняется)
    const tail = /setInterval\(pollFallback, 400\);\s+pollFallback\(\);\s+\}\)\(\);/;
    assert.ok(tail.test(m[1]), 'хвост IIFE (pollFallback) не найден');
    return m[1].replace(tail,
      'setInterval(pollFallback, 400);\n' +
      '    pollFallback();\n' +
      '    globalThis.__T = { renderState, buildMtpBars, get renderedMtpN() { return renderedMtpN; }, setEvents(evts) { clientEventsRev = -1; clientEvents = evts; lastEventsHash = ""; } };\n' +
      '  })();');
  } finally { child.kill('SIGKILL'); }
}

let clientScript = null;
async function getClientScript() {
  if (!clientScript) clientScript = await loadClientScript();
  return clientScript;
}

function mkUiState(draftNMax, posAcc) {
  return {
    mode: 'PLAYBACK', serverOnline: true, is_processing: false, currentStage: 'IDLE',
    n_ctx: 123136, n_prompt_tokens: 0, serverNow: Date.now(), _processingStart: null,
    lastPrefillTps: null, liveCapable: false,
    task: {
      id: null, totalPrompt: 0, kvRestoredTokens: 0, lcpTokens: 0, slotSource: null,
      checkpointRestore: false, reprocessReason: null, prefillTokens: 0, deltaTokens: 0,
      generatedTokens: 0, decodeTps: 0, prefillTps: 0, prefillProgress: 0,
      specAccepted: null, specGenerated: null, specAcceptance: null, specMeanLen: null,
      specInvariantOk: true, isTruncated: false, finalSlotTokens: 0, divergeInfo: null,
      cachedFirstSeen: false,
      // SPEC TOTAL (per-task) — 12 значений: старый баг строил MTP-бары из него
      specPosAcc: ['97', '75', '63', '49', '40', '1', '1', '1', '1', '1', '1', '1']
    },
    ramCache: {
      limitMiB: null, usedMiB: null, promptsCount: null, promptCacheMiB: null,
      cpMax: null, cpCurrent: null, cpCreated: 0, cpErased: 0, cpUsedMiB: 0,
      slots: {}, entries: [], stateAt: null, limitTokens: null, limitTokensEff: null
    },
    specMtp: { posAcc, accTokens: 0, genTokens: 0, meanLen: 0, invariantOk: true },
    ngram: { posAcc: [], accTokens: 0, genTokens: 0, meanLen: 0, invariantOk: true },
    specConfig: { draftNMax },
    serverMetrics: { metricsSeen: false, queueDeferred: 0 },
    eventsRev: 0, events: [], historyRev: 0, history: [],
    player: { idx: 0, total: 0, speed: 1, paused: false, finished: false }
  };
}

function runRenderCase(state, after) {
  const script = clientScript;
  const els = {};
  const makeEl = (id) => ({
    id, innerText: '', innerHTML: '', textContent: '', title: '', value: '',
    checked: false, disabled: false, className: '',
    style: {}, dataset: {}, files: null, scrollTop: 0, scrollHeight: 0,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    addEventListener() {}, removeEventListener() {}, contains: () => false,
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }),
    getContext: () => new Proxy({}, {
      get: (t, p) => (p in t ? t[p] : () => undefined),
      set: (t, p, v) => { t[p] = v; return true; }
    }),
    parentElement: null
  });
  const byId = (id) => els[id] || (els[id] = makeEl(id));
  const keys = ['window', 'location', 'localStorage', 'alert', 'requestAnimationFrame', 'setInterval', 'fetch', 'document'];
  const saved = {};
  for (const k of keys) saved[k] = global[k];
  global.window = global;
  global.location = { search: '' };
  global.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
  global.alert = () => {};
  global.requestAnimationFrame = () => 0;
  global.setInterval = () => 0; // polling/render-циклы не запускаем — детерминизм
  global.fetch = () => new Promise(() => {}); // pollFallback зависает без ошибок
  global.document = {
    getElementById: byId,
    querySelector: () => byId('__q'),
    querySelectorAll: () => [],
    addEventListener() {},
    activeElement: null
  };
  try {
    eval(script);
    const T_ = globalThis.__T;
    assert.ok(T_ && typeof T_.renderState === 'function', '__T.renderState не экспортирован');
    T_.renderState(state);
    if (typeof after === 'function') after(T_); // до restore глобалов (document и др.)
    return { els, T_ };
  } finally {
    for (const k of keys) global[k] = saved[k];
  }
}

const mtpBarCount = (els) => (String(els['mtp-bars'].innerHTML).match(/class="mtp-item"/g) || []).length;

test('UI-render: draftNMax=5 + observed=5 → ровно 5 баров (SPEC TOTAL на 12 позиций игнорируется)', async () => {
  await getClientScript();
  const { els } = runRenderCase(mkUiState(5, ['0.975', '0.698', '0.531', '0.415', '0.337']));
  assert.strictEqual(mtpBarCount(els), 5);
  assert.strictEqual(els['lbl-mtp-pos'].innerText, 'DRAFT-MTP ACC PER POS (1–5):');
  assert.notStrictEqual(els['mtp-v0'].innerText, '—');
});

test('UI-render: draftNMax=8 + observed=8 → ровно 8 баров', async () => {
  await getClientScript();
  const { els } = runRenderCase(mkUiState(8, ['0.975', '0.698', '0.531', '0.415', '0.337', '0.280', '0.240', '0.200']));
  assert.strictEqual(mtpBarCount(els), 8);
  assert.strictEqual(els['lbl-mtp-pos'].innerText, 'DRAFT-MTP ACC PER POS (1–8):');
  assert.notStrictEqual(els['mtp-v7'].innerText, '—');
});

test('UI-render: draftNMax=null (лог без startup-строк) + observed=5 → ровно 5 баров', async () => {
  await getClientScript();
  const { els } = runRenderCase(mkUiState(null, ['0.975', '0.698', '0.531', '0.415', '0.337']));
  assert.strictEqual(mtpBarCount(els), 5);
  assert.strictEqual(els['lbl-mtp-pos'].innerText, 'DRAFT-MTP ACC PER POS (1–5):');
});

test('UI-render: draftNMax=8 + observed=5 → ровно 8 баров, P6–P8 показывают «—»', async () => {
  await getClientScript();
  const { els } = runRenderCase(mkUiState(8, ['0.975', '0.698', '0.531', '0.415', '0.337']));
  assert.strictEqual(mtpBarCount(els), 8);
  assert.strictEqual(els['lbl-mtp-pos'].innerText, 'DRAFT-MTP ACC PER POS (1–8):');
  assert.notStrictEqual(els['mtp-v4'].innerText, '—');
  assert.strictEqual(els['mtp-v5'].innerText, '—');
  assert.strictEqual(els['mtp-v6'].innerText, '—');
  assert.strictEqual(els['mtp-v7'].innerText, '—');
});

test('UI-render: events — серверные ключи рендерятся через i18n (t() не затенён task-объектом)', async () => {
  await getClientScript();
  const st = mkUiState(5, ['0.975', '0.698', '0.531', '0.415', '0.337']);
  st.events = [{ ts: '12:00:00', stage: 'DONE', tag: 'DONE', titleKey: 'ev_done_title', detailKey: 'ev_done_detail', params: { tok: '100', sec: '1.00', tps: '10.0' }, id: 1, taskId: '1' }];
  const { els } = runRenderCase(st, (T_) => {
    T_.setEvents(st.events);
    T_.renderState(st);
  });
  const html = String(els['log-feed'].innerHTML);
  assert.ok(html.includes('Response complete'), 'event title — английский (default lang)');
  assert.ok(html.includes('Generated <b>100</b> tok. in 1.00 s'), 'event detail — params подставлены');
});

// 5: params i18n экранируются в t() — параметр из содержимого лога не может
// инжектить разметку в innerHTML (XSS-замок на join(esc(vars[k]))).
test('UI-render: params i18n экранированы — XSS из параметра не инжектится', async () => {
  await getClientScript();
  const st = mkUiState(5, ['0.975', '0.698', '0.531', '0.415', '0.337']);
  st.events = [{ ts: '12:00:00', stage: 'DONE', tag: 'DONE', titleKey: 'ev_done_title', detailKey: 'ev_done_detail', params: { tok: '<script>alert(1)</script>', sec: '1.00', tps: '10.0' }, id: 1, taskId: '1' }];
  const { els } = runRenderCase(st, (T_) => {
    T_.setEvents(st.events);
    T_.renderState(st);
  });
  const html = String(els['log-feed'].innerHTML);
  assert.ok(!html.includes('<script>alert(1)'), 'сырой XSS из param не попадает в innerHTML');
  assert.ok(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'), 'param экранирован');
});

// 1: --replay не читает stdin: посторонняя строка (task 99) contaminates
// детерминированный снапшот, на который завязаны тесты. rl создаётся только
// в LIVE; снапшот с stdin идентичен снапшоту без stdin.
test('1: --replay игнорирует stdin — посторонняя строка не контаминирует снапшот', () => {
  const lines = [
    { ts: T(0), msg: 'slot operator(): id  0 | task 327 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 100' },
    { ts: T(1), msg: 'slot print_timing: id  0 | task 327 | prompt eval time =  100.00 ms / 100 tokens ( 1.00 ms per token, 1000.00 tokens per second)' },
    { ts: T(2), msg: 'slot print_timing: id  0 | task 327 | n_gen =    10, tg =  20.00 t/s' }
  ];
  const baseline = replay(lines); // stdio: 'ignore' — stdin закрыт
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wwwjs-test-'));
  const log = path.join(dir, 'log.txt');
  const out = path.join(dir, 'out.json');
  fs.writeFileSync(log, lines.map(l => `[${l.ts}] ${l.msg}`).join('\n') + '\n');
  execFileSync('node', [WWW, '--replay', log, out], {
    input: 'slot operator(): id  0 | task 99 | new prompt, n_ctx_slot = 123136, n_keep = 0, task.n_tokens = 777\n',
    stdio: ['pipe', 'ignore', 'ignore'],
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null' }
  });
  const withStdin = JSON.parse(fs.readFileSync(out, 'utf8'));
  assert.deepStrictEqual(withStdin, baseline);
});
