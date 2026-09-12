'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   Самопроверка линтера.

   Проверка не «линтер что-то находит», а «линтер находит ИМЕННО ТУ ошибку,
   из-за которой компилятор TradingView ругался». Поэтому в наборе есть
   образец, воспроизводящий реальный случай: tooltip, разорванный переносом
   через "+", из-за которого пришли CE10017, а до него CE10016.

   Запуск:
       node polygon/test-lint.js
   ═══════════════════════════════════════════════════════════════════════════ */

const fs = require('fs');
const path = require('path');

// lint.js написан для браузера; подставляем минимальный window.
global.window = {};
require(path.join(__dirname, 'lint.js'));
const { lint } = global.window.PineLint;

let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) { pass++; console.log('  OK   ' + name); }
  else { fail++; console.log('  ПРОВАЛ ' + name + (detail ? ' — ' + detail : '')); }
}

const has = (res, rule) => res.issues.some(i => i.rule === rule);

console.log('Линтер Pine — самопроверка\n');

/* ① Реальный случай: tooltip разорван переносом через "+".
   Именно это дало CE10017 на строке 64. */
const broken = [
  '//@version=6',
  'indicator("x")',
  'grp = "1"',
  'useHL = input.bool(false, "Включить", group=grp,',
  '     tooltip="Первая часть текста. " +',
  '     "вторая часть.")',
  'plot(close)',
].join('\n');

let r = lint(broken);
check('ловит перенос строкового выражения (CE10017)',
      has(r, 'CE10017+'), JSON.stringify(r.issues.map(i => i.rule)));

/* ② Незакрытая кавычка */
const unclosed = [
  '//@version=6',
  'indicator("x")',
  'label = "не закрыл',
].join('\n');
r = lint(unclosed);
check('ловит незакрытую строку (CE10017)', has(r, 'CE10017'),
      JSON.stringify(r.issues.map(i => i.rule)));

/* ③ Лишняя закрывающая скобка */
const extraParen = [
  '//@version=6',
  'indicator("x")',
  'plot(close))',
].join('\n');
r = lint(extraParen);
check('ловит лишнюю закрывающую скобку (CE10016)', has(r, 'CE10016'),
      JSON.stringify(r.issues.map(i => i.rule)));

/* ④ Незакрытая скобка в конце файла */
const unclosedParen = [
  '//@version=6',
  'indicator("x")',
  'plot(close',
].join('\n');
r = lint(unclosedParen);
check('ловит незакрытую скобку', has(r, 'скобки'));

/* ⑤ Скобки вокруг условия if */
r = lint(['//@version=6', 'indicator("x")', 'if (close > open)', '    plot(1)'].join('\n'));
check('ловит скобки вокруг if', has(r, 'if(...)'));

/* ⑥ Отсутствие директивы версии */
r = lint(['indicator("x")', 'plot(close)'].join('\n'));
check('ловит отсутствие //@version', has(r, 'version'));

/* ⑦ Табуляция в отступе */
r = lint(['//@version=6', 'indicator("x")', 'if close > open', '\tplot(1)'].join('\n'));
check('ловит табуляцию в отступе', has(r, 'табы'));

/* ⑧ Резервное слово как имя */
r = lint(['//@version=6', 'indicator("x")', 'int = 5'].join('\n'));
check('ловит зарезервированное слово как имя', has(r, 'резерв'));

/* ⑨ Присваивание необъявленной переменной */
r = lint(['//@version=6', 'indicator("x")', 'q := 5'].join('\n'));
check('ловит := для необъявленной', has(r, ':='));

/* ⑩ Правильный код не должен давать ошибок */
const clean = [
  '//@version=6',
  'indicator("x")',
  'len = input.int(20, "Длина")',
  'm = ta.sma(close, len)',
  'plot(m)',
].join('\n');
r = lint(clean);
const errs = r.issues.filter(i => i.severity === 'error');
check('чистый код: ошибок нет', errs.length === 0,
      JSON.stringify(errs.map(i => i.line + ':' + i.rule)));

/* ⑪ Настоящий файл индикатора: ошибок быть не должно */
const pinePath = path.join(__dirname, '..', 'pine', 'mr_quant_v2.pine');
if (fs.existsSync(pinePath)) {
  const real = fs.readFileSync(pinePath, 'utf8');
  r = lint(real);
  const realErrs = r.issues.filter(i => i.severity === 'error');
  check('наш индикатор: ошибок нет (' + r.lines + ' строк)',
        realErrs.length === 0,
        JSON.stringify(realErrs.slice(0, 5).map(i => i.line + ':' + i.rule + ' ' + i.message)));
  if (r.issues.length) {
    console.log('       предупреждений: ' + r.issues.length);
    r.issues.slice(0, 5).forEach(i =>
      console.log('         стр ' + i.line + ' · ' + i.rule + ' · ' + i.message));
  }
} else {
  console.log('  (файл pine/mr_quant_v2.pine не найден — пропускаю)');
}

console.log('\nитого: ' + pass + ' успешно, ' + fail + ' провалено');
process.exit(fail ? 1 : 0);
