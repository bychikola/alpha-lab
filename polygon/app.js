'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   Полигон: связка линтера, виджета TradingView и превью модели.
   ═══════════════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

/* ─────────────────────────── ① Редактор и линтер ─────────────────────── */

function renderIssues(res) {
  const host = $('issues');
  const src = $('editor').value.split('\n');

  if (!res.issues.length) {
    host.innerHTML =
      '<div class="ok"><b>Синтаксических ошибок не найдено</b>' +
      '<div class="sub">Проверено строк: ' + res.lines + '. ' +
      'Типы и версии функций проверит компилятор TradingView.</div></div>';
    return;
  }

  const errs = res.issues.filter(i => i.severity === 'error').length;
  const warns = res.issues.length - errs;

  host.innerHTML =
    '<div class="note" style="border-top:0">Найдено: <b>' + errs + '</b> ошибок, <b>' +
    warns + '</b> предупреждений. Нажмите на строку, чтобы перейти к ней.</div>' +
    res.issues.map((it, idx) =>
      '<div class="issue ' + it.severity + '" data-line="' + it.line + '" data-idx="' + idx + '">' +
        '<div class="ln">' + it.line + '</div>' +
        '<div><div class="msg">' + esc(it.message) + '</div>' +
        '<div class="rule">' + esc(it.rule) + ' · ' + esc(it.title) + '</div></div>' +
        (it.hint ? '<div class="hint">' + esc(it.hint) + '</div>' : '') +
        (it.text ? '<div class="code">' + esc(it.text.slice(0, 160)) + '</div>' : '') +
      '</div>').join('');

  host.querySelectorAll('.issue').forEach(el => {
    el.addEventListener('click', () => selectLine(parseInt(el.dataset.line, 10)));
  });
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/** Ставит курсор на строку и подсвечивает её в textarea. */
function selectLine(line) {
  const ta = $('editor');
  const lines = ta.value.split('\n');
  let start = 0;
  for (let i = 0; i < line - 1 && i < lines.length; i++) start += lines[i].length + 1;
  const end = start + (lines[line - 1] || '').length;
  ta.focus();
  ta.setSelectionRange(start, end);
  // Прокрутка к строке
  const lh = parseFloat(getComputedStyle(ta).lineHeight) || 19;
  ta.scrollTop = Math.max(0, (line - 1) * lh - ta.clientHeight / 2);
}

function runLint() {
  const src = $('editor').value;
  const res = window.PineLint.lint(src);
  renderIssues(res);
  const errs = res.issues.filter(i => i.severity === 'error').length;
  $('stat').textContent = res.lines + ' строк · ' + errs + ' ошибок';
  return res;
}

/* ─────────────────────────── ② Копирование ──────────────────────────── */

function copyCode() {
  const ta = $('editor');
  const text = ta.value;
  const done = () => {
    const b = $('btnCopy');
    const old = b.textContent;
    b.textContent = '✅ Скопировано';
    setTimeout(() => (b.textContent = old), 1600);
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(done, () => fallback(ta, done));
  } else {
    fallback(ta, done);
  }
}

function fallback(ta, done) {
  ta.select();
  try { document.execCommand('copy'); done(); } catch (e) { /* ignore */ }
  ta.setSelectionRange(0, 0);
}

/* ─────────────────────── ③ Виджет TradingView ────────────────────────── */

function mountTV(symbol, interval) {
  const host = $('tv');
  host.textContent = '';

  const container = document.createElement('div');
  container.className = 'tradingview-widget-container';
  container.style.height = '100%';

  const widget = document.createElement('div');
  widget.className = 'tradingview-widget-container__widget';
  widget.style.height = '100%';
  container.appendChild(widget);
  host.appendChild(container);

  // Виджет рисует себя в iframe с tradingview-widget.com. Если страница
  // открыта без сети или iframe заблокирован, панель осталась бы просто
  // пустой — непонятно, грузится она или сломалась. Показываем подсказку,
  // которую iframe перекроет, как только появится.
  widget.innerHTML =
    '<div style="height:100%;display:grid;place-items:center;text-align:center;' +
    'color:#64748b;font-size:12.5px;line-height:1.7;padding:20px">' +
    'Виджет TradingView загружается с tradingview.com.<br>' +
    'Если панель осталась пустой — нет доступа в сеть или браузер ' +
    'блокирует сторонние iframe.<br><br>' +
    '<a href="https://www.tradingview.com/chart/" target="_blank" rel="noopener">' +
    'Открыть график TradingView →</a></div>';

  // Скрипт создаётся через createElement: тег, вставленный через innerHTML,
  // браузер не исполняет. Присваивание s.innerHTML здесь — НЕ работа с HTML:
  // так API виджета TradingView принимает конфигурацию, и содержимое — наш
  // собственный литерал ниже, а не данные извне.
  const s = document.createElement('script');
  s.type = 'text/javascript';
  s.async = true;
  s.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
  s.innerHTML = JSON.stringify({
    autosize: true,
    symbol: symbol,
    interval: interval,
    timezone: 'Etc/UTC',
    theme: 'dark',
    style: '1',
    locale: 'ru',
    backgroundColor: 'rgba(13, 17, 23, 1)',
    gridColor: 'rgba(255, 255, 255, 0.05)',
    hide_side_toolbar: false,
    allow_symbol_change: true,
    save_image: false,
    calendar: false,
    studies: [],
    support_host: 'https://www.tradingview.com',
  });
  container.appendChild(s);
}

/* ─────────────────────────── ④ Превью модели ─────────────────────────── */

const view = { i0: 0, i1: 0 };

function renderPreview() {
  const d = window.ALPHA_PREVIEW;
  if (!d) {
    $('pvNote').innerHTML = 'Нет данных превью. Выполните <b>uv run python ' +
      'scripts/export_preview.py</b> — скрипт создаст <b>polygon/preview.js</b>.';
    return;
  }

  view.i0 = 0;
  view.i1 = d.bars.length - 1;

  $('pvTitle').textContent = d.symbol + ' · ' + d.timeframe + ' · ' +
    d.bars.length + ' баров · окно ' + d.params.window + ', k = ' + d.params.k;

  const s = d.stats;
  const cls = v => v >= 0 ? 'pos' : 'neg';
  $('pvKpis').innerHTML = [
    ['Сделок', s.trades, ''],
    ['Выигрышных', s.wins, ''],
    ['Доля выигрышей', s.win_rate + '%', ''],
    ['Сумма R', (s.sum_r >= 0 ? '+' : '') + s.sum_r, cls(s.sum_r)],
    ['Средний R', (s.avg_r >= 0 ? '+' : '') + s.avg_r, cls(s.avg_r)],
    ['Среднее удержание', s.avg_bars + ' бар', ''],
  ].map(([k, v, c]) =>
    '<div class="kpi"><div class="k">' + k + '</div><div class="v ' + c + '">' +
    v + '</div></div>').join('');

  const stopN = d.trades.filter(t => t.kind === 'стоп').length;
  const takeN = d.trades.filter(t => t.kind === 'тейк').length;
  const timeN = d.trades.filter(t => t.kind === 'время').length;

  $('pvNote').innerHTML =
    'Выходы: <b>' + stopN + '</b> по стопу, <b>' + takeN + '</b> по тейку, <b>' +
    timeN + '</b> по времени. Средний R стопа и тейка показывает, что геометрия ' +
    'модели работает как задумано: −1R и +3R.<br><br>' +
    '<b>Почему это важно.</b> Окно превью короткое, и на нём модель может ' +
    'выглядеть прибыльной. Полный бэктест 2022–2025 по BTCUSDT дал ' +
    '<b>−80.8%</b> при 73.5% капитала, ушедших в комиссии: 144 конфигурации, ' +
    'ни одна не прошла проверку на значимость. Короткий отрезок и длинный ' +
    'период — разные ответы, и верить нужно длинному.';

  draw();
}

function draw() {
  const d = window.ALPHA_PREVIEW;
  if (!d) return;
  window.AlphaChart.drawPrice($('cvPrice'), d, view);
  window.AlphaChart.drawZ($('cvZ'), d, view, d.params.k);
  renderTrades(d);
}

function renderTrades(d) {
  const last = d.trades.slice(-15).reverse();
  // Значения приходят из preview.js, который сгенерирован нашим скриптом из
  // истории Binance, а не вводом пользователя. Всё равно пропускаем через
  // esc(): защита не должна зависеть от того, откуда данные пришли сегодня.
  const rows = last.map((t, i) => {
    const n = d.trades.length - i;
    const dirTxt = t.dir > 0 ? 'лонг' : 'шорт';
    const dirCol = t.dir > 0 ? '#2dd4bf' : '#fb7185';
    const kCol = t.kind === 'стоп' ? '#ef4444' : t.kind === 'тейк' ? '#22c55e' : '#94a3b8';
    const rCol = t.r >= 0 ? '#22c55e' : '#ef4444';
    return '<tr style="border-bottom:1px solid rgba(255,255,255,.06)">' +
      '<td style="padding:8px 10px;color:#64748b" class="mono">' + n + '</td>' +
      '<td style="padding:8px 10px;color:' + dirCol + '">' + esc(dirTxt) + '</td>' +
      '<td style="padding:8px 10px" class="mono">' + esc(t.entry_p) + '</td>' +
      '<td style="padding:8px 10px" class="mono">' + esc(t.exit_p) + '</td>' +
      '<td style="padding:8px 10px;color:' + rCol + '" class="mono">' +
        (t.r >= 0 ? '+' : '') + esc(t.r) + 'R</td>' +
      '<td style="padding:8px 10px;color:' + kCol + '">' + esc(t.kind) + '</td>' +
      '<td style="padding:8px 10px;color:#64748b" class="mono">' + esc(t.bars) + '</td>' +
      '<td style="padding:8px 10px;color:#64748b" class="mono">' +
        esc(String(t.entry_t).slice(0, 16).replace('T', ' ')) + '</td>' +
      '</tr>';
  }).join('');

  $('pvTrades').innerHTML =
    '<thead><tr style="text-align:left;color:#64748b;font-size:10.5px;' +
    'text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid rgba(255,255,255,.1)">' +
    ['№', 'Напр', 'Вход', 'Выход', 'R', 'Причина', 'Бар', 'Время входа']
      .map(h => '<th style="padding:9px 10px;font-weight:600">' + h + '</th>').join('') +
    '</tr></thead><tbody>' + rows + '</tbody>';
}

/* ─────────────────────────── ⑤ Запуск ───────────────────────────────── */

function init() {
  // Код индикатора — из pine-source.js, сгенерированного из pine/mr_quant_v2.pine
  const src = window.PINE_SOURCE || '//@version=6\n// исходник не найден\n';
  $('editor').value = src;

  $('btnLint').addEventListener('click', runLint);
  $('btnCopy').addEventListener('click', copyCode);
  $('btnReset').addEventListener('click', () => {
    $('editor').value = src;
    runLint();
  });

  // Пересчёт при правке — с задержкой, чтобы не считать на каждый символ.
  let t = null;
  $('editor').addEventListener('input', () => {
    clearTimeout(t);
    t = setTimeout(runLint, 600);
  });

  $('tvSymbol').addEventListener('change', () =>
    mountTV($('tvSymbol').value, $('tvInterval').value));
  $('tvInterval').addEventListener('change', () =>
    mountTV($('tvSymbol').value, $('tvInterval').value));

  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 150); });

  runLint();
  renderPreview();
  mountTV($('tvSymbol').value, $('tvInterval').value);
}

document.addEventListener('DOMContentLoaded', init);
