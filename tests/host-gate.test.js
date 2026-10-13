'use strict';
// 3: Host gate (DNS rebinding defense).
//   - default (loopback bind): allowlist = loopback names; foreign Host -> 403
//     on EVERY route (loopback bind alone does not stop rebinding: the browser
//     on evil.example sends Host: evil.example and reaches 127.0.0.1);
//   - MONITOR_ALLOWED_HOSTS extends the list and makes it strict;
//   - wildcard bind (MONITOR_HOST=0.0.0.0) without an allowlist: any Host is
//     accepted (LAN deployments preserved) + startup warning.
// Запуск: node --test tests/
const { test } = require('node:test');
const assert = require('node:assert');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');

const WWW = path.join(__dirname, '..', 'www.js');
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function startMonitor(env = {}) {
  const monPort = 40000 + Math.floor(Math.random() * 9000);
  let stderr = '', stdout = '';
  const child = spawn('node', [WWW, '--port', String(monPort), '--url', 'http://127.0.0.1:1'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, LLAMA_LOG_FILE: '/dev/null', ...env }
  });
  child.stderr.on('data', d => { stderr += d; if (stderr.length > 8000) stderr = stderr.slice(-8000); });
  child.stdout.on('data', d => { stdout += d; if (stdout.length > 8000) stdout = stdout.slice(-8000); });
  let up = false;
  for (let i = 0; i < 100 && !up; i++) {
    await sleep(100);
    try { up = (await fetch(`http://127.0.0.1:${monPort}/api/state`)).ok; } catch (_) { /* ещё стартует */ }
  }
  assert.ok(up, 'monitor is up (stderr: ' + stderr + ')');
  return { child, monPort, getStderr: () => stderr, getStdout: () => stdout };
}

// Запрос на 127.0.0.1, но с произвольным Host-заголовком (как у браузера
// с чужого домена / curl -H 'Host: ...').
function reqWithHost(monPort, hostHeader, p = '/api/state') {
  return new Promise((resolve, reject) => {
    const req = http.request({ host: '127.0.0.1', port: monPort, path: p, headers: { Host: hostHeader } }, (res) => {
      let b = '';
      res.on('data', d => { b += d; });
      res.on('end', () => resolve({ status: res.statusCode, body: b }));
    });
    req.on('error', reject);
    req.end();
  });
}

test('3: foreign Host -> 403 on every route; loopback hosts pass (any port)', async () => {
  const { child, monPort, getStderr } = await startMonitor();
  try {
    for (const p of ['/', '/api/state', '/events', '/favicon.ico']) {
      const r = await reqWithHost(monPort, 'evil.example', p);
      assert.strictEqual(r.status, 403, `Host: evil.example на ${p} должен быть 403`);
    }
    // Host с портом — сравнение по hostname (любой порт)
    let r = await reqWithHost(monPort, 'evil.example:3000');
    assert.strictEqual(r.status, 403);
    // loopback-имена проходят
    for (const h of ['127.0.0.1', 'localhost']) {
      r = await reqWithHost(monPort, `${h}:${monPort}`);
      assert.strictEqual(r.status, 200, `Host: ${h} должен проходить`);
    }
  } finally {
    child.kill('SIGKILL');
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});

test('3: MONITOR_ALLOWED_HOSTS расширяет allowlist и делает его строгим', async () => {
  const { child, monPort, getStderr } = await startMonitor({ MONITOR_ALLOWED_HOSTS: 'evil.example, monitor.internal' });
  try {
    let r = await reqWithHost(monPort, 'evil.example');
    assert.strictEqual(r.status, 200, 'объявленный хост проходит');
    r = await reqWithHost(monPort, 'monitor.internal');
    assert.strictEqual(r.status, 200, 'второй объявленный хост проходит');
    r = await reqWithHost(monPort, 'other.example');
    assert.strictEqual(r.status, 403, 'необъявленный хост остаётся 403');
    r = await reqWithHost(monPort, `127.0.0.1:${monPort}`);
    assert.strictEqual(r.status, 200, 'loopback остаётся разрешён');
  } finally {
    child.kill('SIGKILL');
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});

test('3: wildcard bind без allowlist — любой Host + предупреждение при старте', async () => {
  const { child, monPort, getStdout, getStderr } = await startMonitor({ MONITOR_HOST: '0.0.0.0' });
  try {
    const r = await reqWithHost(monPort, 'anything.example');
    assert.strictEqual(r.status, 200, 'явная LAN-развёртка (0.0.0.0) сохраняет доступ');
    assert.match(getStdout(), /MONITOR_ALLOWED_HOSTS/, 'стартовое предупреждение об открытом Host');
  } finally {
    child.kill('SIGKILL');
    if (child.exitCode !== 0) console.error('monitor stderr:', getStderr());
  }
});
