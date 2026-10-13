'use strict';
// Статические замки на правки ревью (SPEC: не давать классу дефекта вернуться):
//  4 — mermaid-скрипт с integrity (sha384) + crossorigin;
//  6 — панели .status-bar/.player-toolbar не скрыты display:none !important
//      (единственный механизм скрытия был CSS — JS их не трогает).
// Функциональный замок на экранирование params в t() — в replay.test.js
// (UI-render: params i18n экранированы).
// Запуск: node --test tests/
const { test } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const SRC = fs.readFileSync(path.join(__dirname, '..', 'www.js'), 'utf8');

test('4: mermaid script carries integrity (sha384) + crossorigin', () => {
  const m = SRC.match(/<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/mermaid@[^"]*"[^>]*>/);
  assert.ok(m, 'mermaid script tag найден');
  assert.match(m[0], /integrity="sha384-[A-Za-z0-9+/]+={0,3}"/, 'integrity sha384 присутствует');
  assert.match(m[0], /crossorigin="anonymous"/, 'crossorigin присутствует (иначе integrity не применяется)');
});

// Правила .log-row.* используют display: none !important легитимно (фильтры
// ленты) — проверяем конкретно базовые правила панелей.
test('6: .status-bar/.player-toolbar base rules are not hidden by display:none', () => {
  for (const sel of ['.status-bar', '.player-toolbar']) {
    const m = SRC.match(new RegExp('\\' + sel + ' \\{[^}]*\\}'));
    assert.ok(m, sel + ': базовое правило найдено');
    assert.ok(!/display:\s*none/.test(m[0]), sel + ' не должен прятаться CSS-оверрайдом');
  }
});

// Замки пакета i18n/SSE (2026-10-13): дефекты, найденные ревьюером в headless
// Chromium (смена языка ломала mermaid-легенду и не перерисовывала журнал;
// onopen пинил pollFallback без единого snapshot).
test('7: applyI18n resets data-processed before re-running mermaid', () => {
  const m = SRC.match(/const pre = document\.querySelector[\s\S]*?mermaid\.run\(\)/);
  assert.ok(m, 'блок mermaid в applyI18n найден');
  assert.match(m[0], /pre\.removeAttribute\('data-processed'\)/,
    'без сброса data-processed Mermaid пропускает уже отрисованный элемент при следующем run()');
});

test('8: lang-toggle resets render caches and re-renders', () => {
  const m = SRC.match(/langToggle\.addEventListener\('click', \(\) => \{[\s\S]*?\n    \}\);/);
  assert.ok(m, 'обработчик lang-toggle найден');
  for (const k of ['lastEventsHash', 'lastHistHash', 'lastLiveHash']) {
    assert.ok(m[0].includes(k + " = ''"), k + ' сбрасывается');
  }
  assert.match(m[0], /if \(latestState\) scheduleRender\(latestState\)/, 'перерисовка запрошена');
});

test('9: es.onopen does not claim sseWorking', () => {
  const m = SRC.match(/es\.onopen = \(\) => \{[^}]*\};/);
  assert.ok(m, 'es.onopen найден');
  assert.ok(!/sseWorking\s*=\s*true/.test(m[0]),
    'onopen срабатывает до данных; sseWorking=true — только в onmessage');
});

test('10: mermaid_not_rendered key exists in both dictionaries', () => {
  const hits = SRC.match(/mermaid_not_rendered: '[^']*'/g) || [];
  assert.strictEqual(hits.length, 2, 'ключ ровно в двух словарях (en + ru)');
  assert.ok(hits.some(h => /not rendered yet/.test(h)), 'en-текст');
  assert.ok(hits.some(h => /ещё не отрисована/.test(h)), 'ru-текст');
});
