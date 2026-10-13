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
