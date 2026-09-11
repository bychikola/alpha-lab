'use strict';

/* Alpha Lab — дашборд статического отчёта (задача 19).
 *
 * Работает по file://: данные приходят из report.js как window.ALPHA_REPORT,
 * потому что fetch() к локальным файлам браузер блокирует CORS-политикой.
 * Ни сборки, ни npm, ни внешних JS-библиотек — только Canvas.
 *
 * Контракт: schema_version, verdict, series, extra (см. src/alpha_lab/report/writer.py).
 * Неизвестная мажорная версия схемы — отказ от рендера с видимым сообщением:
 * частично совместимые графики выглядели бы правдоподобно, но были бы неверны.
 */

const REQUIRED_MAJOR = 1;

const MONO_FONT = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const C = {
  text: '#cbd5e1', dim: '#64748b', grid: 'rgba(255,255,255,.05)',
  sky: '#38bdf8', red: '#ef4444', green: '#22c55e', amber: '#f59e0b', slate: '#cbd5e1'
};

/* ── форматирование ─────────────────────────────────────────── */

const ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
function esc(s) { return String(s).replace(/[&<>"']/g, ch => ESC_MAP[ch]); }

function isNum(v) { return v !== null && v !== undefined && isFinite(v); }

function fmtNum(v, digits = 2) {
  if (!isNum(v)) return '—';
  return Number(v).toLocaleString('ru-RU', {
    minimumFractionDigits: digits, maximumFractionDigits: digits
  });
}

function fmtPct(v, digits = 1) {
  if (!isNum(v)) return '—';
  return (Number(v) * 100).toLocaleString('ru-RU', {
    minimumFractionDigits: digits, maximumFractionDigits: digits
  }) + '%';
}

function fmtInt(v) {
  if (!isNum(v)) return '—';
  return Math.round(Number(v)).toLocaleString('ru-RU');
}

function fmtPrice(v) {
  if (!isNum(v)) return '—';
  return Number(v).toLocaleString('ru-RU', {
    minimumFractionDigits: 2, maximumFractionDigits: 2
  });
}

function fmtGenerated(iso) {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
  });
}

function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  if (a > 10 && a < 20) return many;
  if (b > 1 && b < 5) return few;
  if (b === 1) return one;
  return many;
}

/* ── состояния отказа ───────────────────────────────────────── */

function renderError(title, bodyHtml) {
  document.getElementById('app').innerHTML =
    `<div class="error"><h2>${esc(title)}</h2>${bodyHtml}</div>`;
  const meta = document.getElementById('top-meta');
  if (meta) meta.textContent = 'отчёт не отображён';
}

function majorOf(version) {
  const m = /^\s*(\d+)\s*\./.exec(String(version));
  return m ? parseInt(m[1], 10) : NaN;
}

function structuralProblem(r) {
  if (!r.verdict || typeof r.verdict !== 'object') {
    return 'В отчёте нет блока <code>verdict</code> — рисовать нечего.';
  }
  if (!r.series || typeof r.series !== 'object') {
    return 'В отчёте нет блока <code>series</code> — рисовать нечего.';
  }
  const s = r.series;
  if (!Array.isArray(s.ts) || s.ts.length < 2) {
    return 'Ряд <code>series.ts</code> отсутствует или короче двух точек.';
  }
  if (!Array.isArray(s.equity) || s.equity.length !== s.ts.length) {
    return 'Ряд <code>series.equity</code> отсутствует или не совпадает по длине с <code>series.ts</code>. ' +
           'Контракт требует выровненных рядов одной длины — иначе график был бы правдоподобным, но неверным.';
  }
  for (const key of ['drawdown', 'close', 'position', 'fee', 'slippage', 'funding']) {
    const v = s[key];
    if (v !== undefined && (!Array.isArray(v) || v.length !== s.ts.length)) {
      return `Ряд <code>series.${key}</code> не совпадает по длине с <code>series.ts</code>.`;
    }
  }
  return null;
}

/* ── метки времени ──────────────────────────────────────────── */

/* writer берёт ts из индекса equity. Если индекс — RangeIndex, pd.Timestamp(int)
 * даёт 1970-01-01T00:00:00.000000001 и т.п.: формально ISO, фактически мусор.
 * Поэтому неправдоподобные метки (год < 2000) не показываем — ось идёт по номерам баров. */
function parseTimestamps(ts) {
  if (!Array.isArray(ts) || ts.length < 2) return null;
  const first = Date.parse(ts[0]);
  const last = Date.parse(ts[ts.length - 1]);
  if (!isFinite(first) || !isFinite(last)) return null;
  if (new Date(first).getUTCFullYear() < 2000) return null;
  if (last < first) return null;
  return ts.map(t => { const v = Date.parse(t); return isFinite(v) ? v : null; });
}

function fmtDateShort(ms) {
  const d = new Date(ms), p = x => String(x).padStart(2, '0');
  return `${p(d.getUTCDate())}.${p(d.getUTCMonth() + 1)}.${String(d.getUTCFullYear()).slice(2)}`;
}

function fmtDateTime(ms) {
  const d = new Date(ms), p = x => String(x).padStart(2, '0');
  return `${p(d.getUTCDate())}.${p(d.getUTCMonth() + 1)}.${d.getUTCFullYear()} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;
}

function timeMarks(n, tsMs, count) {
  const out = [];
  if (n < 2) return out;
  for (let k = 0; k < count; k++) {
    const i = Math.round((k * (n - 1)) / (count - 1));
    out.push({ i, text: tsMs && tsMs[i] !== null ? fmtDateShort(tsMs[i]) : 'бар ' + fmtInt(i) });
  }
  return out;
}

/* ── вердикт ────────────────────────────────────────────────── */

function kpiCard(m) {
  const cls = m.pass === null || m.pass === undefined ? 'na' : (m.pass ? 'pass' : 'fail');
  const fill = isNum(m.fill) ? Math.max(0, Math.min(1, m.fill)) : 0;
  return `<div class="kpi ${cls}">
    <div class="k">${esc(m.label)}</div>
    <div class="v mono">${esc(m.value)}</div>
    <div class="t">${esc(m.hint)}</div>
    <div class="bar"><i style="width:${(fill * 100).toFixed(1)}%"></i></div>
  </div>`;
}

const METRIC_LABELS = {
  sharpe: 'Sharpe', sortino: 'Sortino', max_dd: 'Макс. просадка', calmar: 'Calmar',
  profit_factor: 'Профит-фактор', total_return: 'Доходность', trades: 'Сделки',
  win_rate: 'Winrate', avg_trade: 'Средняя сделка'
};
const PCT_METRICS = { max_dd: 1, total_return: 1, win_rate: 1, avg_trade: 1 };
const CORE_METRICS = { sharpe: 1, max_dd: 1, total_return: 1, trades: 1 };

function renderVerdict(report) {
  const v = report.verdict;
  const extra = report.extra || {};
  const alive = !!v.alive;
  /* P5: черновой вердикт — третий исход. Прохождение screening-гейтов делает
   * конфигурацию кандидатом, а не «живой» и не «мёртвой»: показать её как
   * МЕРТВА значило бы выдать непроверенное за отвергнутое. */
  const rough = !!v.screening;

  const dsr = v.dsr, p = v.p_value, pbo = v.pbo, trades = v.trades;
  const kpis = [
    { label: 'DSR', value: fmtNum(dsr, 4), hint: 'порог ≥ 0,95',
      pass: isNum(dsr) ? dsr >= 0.95 : null, fill: isNum(dsr) ? dsr / 0.95 : null },
    { label: 'p-value', value: fmtNum(p, 4), hint: 'порог < 0,05',
      pass: isNum(p) ? p < 0.05 : null, fill: isNum(p) ? p / 0.05 : null },
    { label: 'PBO', value: isNum(pbo) ? fmtNum(pbo, 2) : '—',
      hint: isNum(pbo) ? 'порог < 0,50'
        : 'недоступно (нужна матрица доходностей конфигураций)',
      pass: isNum(pbo) ? pbo < 0.5 : null, fill: isNum(pbo) ? pbo / 0.5 : null },
    { label: 'Сделок', value: fmtInt(trades), hint: 'минимум 100',
      pass: isNum(trades) ? trades >= 100 : null, fill: isNum(trades) ? trades / 100 : null },
    { label: 'Sharpe', value: fmtNum(v.sharpe), hint: 'годовая шкала', pass: null, fill: null },
    { label: 'Max DD', value: fmtPct(v.max_dd), hint: 'от пика equity', pass: null, fill: null },
    { label: 'Доходность', value: fmtPct(v.total_return), hint: 'за прогон', pass: null, fill: null },
    { label: 'Попыток', value: fmtInt(v.n_configs_tried), hint: 'учтено в DSR', pass: null, fill: null }
  ].map(kpiCard).join('');

  const reasons = Array.isArray(v.reasons) ? v.reasons : [];
  const reasonsHtml = reasons.length
    ? `<ul class="reasons ${alive ? 'ok' : ''}">` +
      reasons.map(r => `<li>${esc(r)}</li>`).join('') + '</ul>'
    : '';

  /* Предупреждения —amber-панель, НЕ красный список причин: гейт не пройден
   * не потому, что стратегия плоха, а потому, что входных данных для проверки
   * нет (одиночная конфигурация, нет funding, разрывы). Смешать их с reasons
   * значило бы или ложно убить стратегию, или спрятать пробел в проверке. */
  const warnings = Array.isArray(v.warnings)
    ? v.warnings.filter(w => typeof w === 'string' && w.length > 0)
    : [];
  const warningsHtml = warnings.length
    ? `<div class="alert warn">
         <h3>Предупреждения — эти гейты не проверены</h3>
         ${warnings.map(w => `<p>${esc(w)}</p>`).join('')}
       </div>`
    : '';

  const metrics = (v.metrics && typeof v.metrics === 'object') ? v.metrics : {};
  const metricsHtml = Object.keys(metrics)
    .filter(k => !CORE_METRICS[k] && metrics[k] !== null && metrics[k] !== undefined)
    .map(k => {
      const val = PCT_METRICS[k] ? fmtPct(metrics[k], 2) : fmtNum(metrics[k], 4);
      return `<div class="metric"><div class="mk">${esc(METRIC_LABELS[k] || k)}</div>
              <div class="mv mono">${esc(val)}</div></div>`;
    }).join('');

  const chips = [];
  if (rough) {
    chips.push(`<span class="chip amber" title="Черновой прогон (screening): меньше перестановок, минимальный достижимый p-value 1/(N+1). Alive по черновому вердикту не выносится.">черновой вердикт · перестановок ${fmtInt(v.n_permutations)}</span>`);
  }
  if (extra.symbol) chips.push(`<span class="chip mono">${esc(extra.symbol)}${extra.timeframe ? ' · ' + esc(extra.timeframe) : ''}</span>`);
  if (extra.data_version) chips.push(`<span class="chip mono" title="версия набора данных">данные ${esc(extra.data_version)}</span>`);
  const gen = fmtGenerated(report.generated_at);
  if (gen) chips.push(`<span class="chip">отчёт от ${esc(gen)}</span>`);
  if (isNum(extra.dirty_bars) && extra.dirty_bars > 0) {
    chips.push(`<span class="chip amber" title="Стратегия на грязных барах не торгует; они исключены, а не интерполированы.">грязных баров: ${fmtInt(extra.dirty_bars)}</span>`);
  }
  const cap = extra.capacity;
  if (cap && cap.over_capacity) {
    chips.push(`<span class="chip red">ёмкость: превышена</span>`);
  } else if (cap && isNum(cap.max_participation_observed) && isNum(cap.max_participation_limit)) {
    chips.push(`<span class="chip green" title="Заявки не превышали долю объёма бара.">ёмкость: ok (макс. ${fmtPct(cap.max_participation_observed, 4)} при лимите ${fmtPct(cap.max_participation_limit, 2)})</span>`);
  }

  // «Жива» с непроверенными гейтами — не то же самое, что «жива» после всех
  // проверок; подпись обязана это различать. Черновое прохождение — кандидат,
  // а не результат: alive по screening не выносится.
  const candidate = rough && !alive && reasons.length === 0;
  const statusNote = alive
    ? (warnings.length
        ? 'прошла проверку с оговорками — часть гейтов не проверена, см. предупреждения'
        : 'прошла проверку на значимость и переобучение')
    : candidate
      ? 'черновой вердикт (screening): гейты пройдены, но alive не вынесен — нужен полный прогон'
      : 'не прошла проверку — причины ниже';
  const statusText = alive ? 'ЖИВА' : (candidate ? 'КАНДИДАТ' : 'МЕРТВА');

  document.getElementById('verdict-box').innerHTML =
    `<div class="verdict ${alive ? 'alive' : 'dead'}">
       <div class="verdict-top">
         <div>
           <h1>${statusText}</h1>
           <div class="status-note">${statusNote}</div>
           <div class="sub">
             <span class="mono">${esc(v.strategy_name)}</span> ·
             experiment <span class="mono">${esc(v.experiment_id)}</span>
           </div>
         </div>
         <div class="verdict-meta">${chips.join('')}</div>
       </div>
       ${warningsHtml}
       <div class="kpis">${kpis}</div>
       ${reasonsHtml}
       ${metricsHtml ? `<div class="metrics-strip">${metricsHtml}</div>` : ''}
     </div>`;
}

/* ── предупреждения extra ───────────────────────────────────── */

function capacityBanner(extra) {
  const cap = extra && extra.capacity;
  if (!cap || !cap.over_capacity) return '';
  const hits = isNum(cap.cap_hits) ? cap.cap_hits : null;
  const hitsTxt = hits === null ? 'Часть заявок'
    : `${fmtInt(hits)} ${plural(hits, 'заявка', 'заявки', 'заявок')}`;
  // «21 заявка превысила», но «11 заявок превысили» — согласуем и глагол.
  const verb = hits === null || (hits % 10 === 1 && hits % 100 !== 11) ? 'превысила' : 'превысили';
  const observed = isNum(cap.max_participation_observed)
    ? fmtPct(cap.max_participation_observed, 4) : '—';
  const limit = isNum(cap.max_participation_limit)
    ? fmtPct(cap.max_participation_limit, 2) : '—';
  return `<div class="alert warn">
    <h3>Ёмкость не доказана — прогон оптимистичен</h3>
    <p>${esc(hitsTxt)} ${verb} лимит участия в объёме бара: максимум
       <b class="mono">${esc(observed)}</b> при лимите <b class="mono">${esc(limit)}</b>.
       Движок не режет и не переносит заявки, поэтому смоделированные издержки
       могут быть занижены, а результат — мягче реальности.</p>
  </div>`;
}

/* ── canvas: общие примитивы ────────────────────────────────── */

function chartBox(w, h) {
  return { x0: 62, x1: Math.max(90, w - 12), y0: 10, y1: Math.max(30, h - 22) };
}

function prepareDraw(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(140, canvas.clientWidth | 0);
  const h = Math.max(60, canvas.clientHeight | 0);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function extent(values, include) {
  let min = Infinity, max = -Infinity, found = false;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null || v === undefined || !isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    found = true;
  }
  if (isNum(include)) { min = Math.min(min, include); max = Math.max(max, include); }
  if (!found) return null;
  if (max - min < 1e-12) max = min + (Math.abs(min) || 1) * 0.01;
  const pad = (max - min) * 0.06;
  return { min: min - pad, max: max + pad };
}

function niceTicks(min, max, count) {
  const span = max - min;
  if (!isFinite(span) || span <= 0) return [min];
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9 && out.length < 24; v += step) out.push(v);
  return out;
}

function drawEmpty(ctx, w, h) {
  ctx.fillStyle = C.dim;
  ctx.font = '12px ' + MONO_FONT;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('нет данных', w / 2, h / 2);
}

function drawFrame(ctx, box, ticks, yOf, yFmt, xMarks, n, xOf) {
  ctx.save();
  ctx.font = '10px ' + MONO_FONT;
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  ctx.fillStyle = C.dim;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const t of ticks) {
    const y = Math.round(yOf(t)) + 0.5;
    if (y < box.y0 - 1 || y > box.y1 + 1) continue;
    ctx.beginPath(); ctx.moveTo(box.x0, y); ctx.lineTo(box.x1, y); ctx.stroke();
    ctx.fillText(yFmt(t), box.x0 - 8, y);
  }
  ctx.textBaseline = 'top';
  for (const m of xMarks) {
    const x = Math.max(box.x0, Math.min(box.x1, xOf(m.i)));
    ctx.textAlign = m.i === 0 ? 'left' : (m.i === n - 1 ? 'right' : 'center');
    ctx.fillText(m.text, x, box.y1 + 7);
  }
  ctx.restore();
}

function drawLine(ctx, box, values, n, xOf, yOf, color, width) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  ctx.beginPath();
  let open = false;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v === null || v === undefined || !isFinite(v)) { open = false; continue; }
    const x = xOf(i), y = yOf(v);
    if (!open) { ctx.moveTo(x, y); open = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawArea(ctx, box, values, n, xOf, yOf, baseY, color) {
  ctx.save(); ctx.fillStyle = color;
  let run = [];
  const flush = () => {
    if (run.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(xOf(run[0][0]), yOf(run[0][1]));
      for (const p of run) ctx.lineTo(xOf(p[0]), yOf(p[1]));
      ctx.lineTo(xOf(run[run.length - 1][0]), baseY);
      ctx.lineTo(xOf(run[0][0]), baseY);
      ctx.closePath(); ctx.fill();
    }
    run = [];
  };
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v === null || v === undefined || !isFinite(v)) { flush(); continue; }
    run.push([i, v]);
  }
  flush();
  ctx.restore();
}

/* Диапазон high–low: агрегируем по пиксельным колонкам, иначе 35k прямоугольников. */
function drawRangeBand(ctx, box, hi, lo, n, yOf, color) {
  const cols = Math.max(2, Math.round(box.x1 - box.x0));
  const top = new Float64Array(cols).fill(-Infinity);
  const bot = new Float64Array(cols).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const h = hi[i], l = lo[i];
    if (h === null || l === null || h === undefined || l === undefined) continue;
    if (!isFinite(h) || !isFinite(l)) continue;
    const c = Math.min(cols - 1, Math.floor((i / Math.max(1, n - 1)) * (cols - 1) + 1e-9));
    if (h > top[c]) top[c] = h;
    if (l < bot[c]) bot[c] = l;
  }
  const xc = c => box.x0 + (c / (cols - 1)) * (box.x1 - box.x0);
  ctx.save(); ctx.fillStyle = color;
  let start = -1;
  const flush = end => {
    if (start < 0 || end < start) { start = -1; return; }
    ctx.beginPath();
    ctx.moveTo(xc(start), yOf(top[start]));
    for (let c = start + 1; c <= end; c++) ctx.lineTo(xc(c), yOf(top[c]));
    for (let c = end; c >= start; c--) ctx.lineTo(xc(c), yOf(bot[c]));
    ctx.closePath(); ctx.fill();
    start = -1;
  };
  for (let c = 0; c < cols; c++) {
    if (isFinite(top[c]) && isFinite(bot[c])) { if (start < 0) start = c; }
    else flush(c - 1);
  }
  flush(cols - 1);
  ctx.restore();
}

function posKind(v) { return v > 0 ? 1 : (v < 0 ? -1 : 0); }

/* Фоновые полосы позиции: long — зелёные, short — красные. */
function drawPositionBands(ctx, box, pos, n, xOf, colors, y0, y1) {
  let start = 0;
  for (let i = 1; i <= n; i++) {
    const prev = posKind(pos[start]);
    const cur = i < n ? posKind(pos[i]) : 0;
    if (i === n || cur !== prev) {
      if (prev !== 0) {
        const x0 = xOf(start), x1 = xOf(i - 1);
        ctx.fillStyle = colors[prev];
        ctx.fillRect(x0, y0, Math.max(1, x1 - x0), y1 - y0);
      }
      start = i;
    }
  }
}

function drawHoverLine(ctx, box, x) {
  ctx.save();
  ctx.strokeStyle = 'rgba(226,232,240,.4)';
  ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(Math.round(x) + 0.5, box.y0);
  ctx.lineTo(Math.round(x) + 0.5, box.y1);
  ctx.stroke();
  ctx.restore();
}

/* ── реестр графиков (перерисовка на resize и при наведении) ── */

const registry = [];

function registerChart(canvas, n, readoutEl, defaultText, hoverText, draw) {
  const entry = { canvas, n, readoutEl, defaultText, hoverText, draw, hover: -1 };
  registry.push(entry);
  readoutEl.innerHTML = defaultText;
  canvas.addEventListener('mousemove', ev => {
    const rect = canvas.getBoundingClientRect();
    const box = chartBox(rect.width, rect.height);
    const x = ev.clientX - rect.left;
    if (x < box.x0 - 8 || x > box.x1 + 8) { setHover(entry, -1); return; }
    const t = (x - box.x0) / Math.max(1, box.x1 - box.x0);
    setHover(entry, Math.max(0, Math.min(n - 1, Math.round(t * (n - 1)))));
  });
  canvas.addEventListener('mouseleave', () => setHover(entry, -1));
  return entry;
}

function setHover(entry, i) {
  if (entry.hover === i) return;
  entry.hover = i;
  entry.draw(i);
  entry.readoutEl.innerHTML = i < 0 ? entry.defaultText : entry.hoverText(i);
}

function redrawAll() {
  for (const e of registry) e.draw(e.hover);
}

/* ── три графика ────────────────────────────────────────────── */

function drawEquityChart(canvas, values, n, tsMs, hover) {
  const { ctx, w, h } = prepareDraw(canvas);
  const box = chartBox(w, h);
  let base = null;
  for (let i = 0; i < n; i++) {
    if (values[i] !== null && isFinite(values[i])) { base = values[i]; break; }
  }
  const ext = extent(values, base);
  if (!ext) { drawEmpty(ctx, w, h); return; }
  const yOf = v => box.y1 - ((v - ext.min) / (ext.max - ext.min)) * (box.y1 - box.y0);
  const xOf = i => box.x0 + (i / Math.max(1, n - 1)) * (box.x1 - box.x0);
  const marks = timeMarks(n, tsMs, 5);
  drawFrame(ctx, box, niceTicks(ext.min, ext.max, 4), yOf, v => fmtNum(v, 2), marks, n, xOf);
  drawArea(ctx, box, values, n, xOf, yOf, box.y1, 'rgba(56,189,248,.13)');
  if (isNum(base)) {
    ctx.save();
    ctx.strokeStyle = 'rgba(148,163,184,.35)'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(box.x0, yOf(base)); ctx.lineTo(box.x1, yOf(base)); ctx.stroke();
    ctx.restore();
  }
  drawLine(ctx, box, values, n, xOf, yOf, C.sky, 1.6);
  if (hover >= 0) drawHoverLine(ctx, box, xOf(hover));
}

function drawDrawdownChart(canvas, values, n, tsMs, hover) {
  const { ctx, w, h } = prepareDraw(canvas);
  const box = chartBox(w, h);
  const ext = extent(values, 0);
  if (!ext) { drawEmpty(ctx, w, h); return; }
  const yOf = v => box.y1 - ((v - ext.min) / (ext.max - ext.min)) * (box.y1 - box.y0);
  const xOf = i => box.x0 + (i / Math.max(1, n - 1)) * (box.x1 - box.x0);
  const marks = timeMarks(n, tsMs, 5);
  drawFrame(ctx, box, niceTicks(ext.min, ext.max, 3), yOf, v => fmtPct(v, 0), marks, n, xOf);
  drawArea(ctx, box, values, n, xOf, yOf, yOf(0), 'rgba(239,68,68,.18)');
  drawLine(ctx, box, values, n, xOf, yOf, C.red, 1.3);
  if (hover >= 0) drawHoverLine(ctx, box, xOf(hover));
}

function drawPriceChart(canvas, s, n, tsMs, hover) {
  const { ctx, w, h } = prepareDraw(canvas);
  const box = chartBox(w, h);
  const hi = Array.isArray(s.high) ? s.high : null;
  const lo = Array.isArray(s.low) ? s.low : null;
  let ext = hi && lo ? extent(hi.concat(lo), null) : extent(s.close, null);
  if (!ext) { drawEmpty(ctx, w, h); return; }

  const yOf = v => box.y1 - ((v - ext.min) / (ext.max - ext.min)) * (box.y1 - box.y0);
  const xOf = i => box.x0 + (i / Math.max(1, n - 1)) * (box.x1 - box.x0);
  const marks = timeMarks(n, tsMs, 5);
  drawFrame(ctx, box, niceTicks(ext.min, ext.max, 4), yOf, v => fmtPrice(v), marks, n, xOf);

  if (Array.isArray(s.position)) {
    drawPositionBands(ctx, box, s.position, n, xOf,
      { 1: 'rgba(34,197,94,.07)', '-1': 'rgba(239,68,68,.07)' }, box.y0, box.y1);
  }
  if (hi && lo) drawRangeBand(ctx, box, hi, lo, n, yOf, 'rgba(148,163,184,.14)');
  drawLine(ctx, box, s.close, n, xOf, yOf, C.slate, 1.4);
  if (Array.isArray(s.position)) {
    // Лента позиции внизу: сразу видно, где стратегия была в рынке.
    drawPositionBands(ctx, box, s.position, n, xOf,
      { 1: 'rgba(34,197,94,.8)', '-1': 'rgba(239,68,68,.8)' },
      box.y1 - 9, box.y1);
  }
  if (hover >= 0) drawHoverLine(ctx, box, xOf(hover));
}

/* ── издержки ───────────────────────────────────────────────── */

function renderCosts(report) {
  const extra = report.extra || {};
  const totals = extra.costs_total || {};
  // Только явный false означает «данных о funding нет»; отсутствие поля —
  // старый отчёт, где считаем поведение прежним (нулевой funding как факт).
  const fundingAvailable = extra.funding_available !== false;
  const eq0 = (() => {
    const eq = report.series.equity;
    for (let i = 0; i < eq.length; i++) if (isNum(eq[i])) return eq[i];
    return 1;
  })();
  const denom = Math.abs(eq0) > 1e-12 ? eq0 : 1;

  // null в значении Funding рендерится как «н/д», а не как 0: иначе панель
  // утверждала бы, что финансирование не стоило ничего, тогда как данных о нём
  // просто нет, и издержки занижены.
  const rows = [
    ['Комиссии', isNum(totals.fee) ? Number(totals.fee) : 0],
    ['Проскальзывание', isNum(totals.slippage) ? Number(totals.slippage) : 0],
    ['Funding', fundingAvailable
      ? (isNum(totals.funding) ? Number(totals.funding) : 0) : null]
  ];
  const sum = rows.reduce((a, r) => a + (r[1] === null ? 0 : r[1]), 0);
  const absSum = rows.reduce((a, r) => a + Math.abs(r[1] === null ? 0 : r[1]), 0) || 1;

  const body = rows.map(([label, val]) => {
    if (val === null) {
      const events = isNum(extra.funding_events) ? fmtInt(extra.funding_events) : '—';
      return `<tr title="Funding недоступен: файл отсутствует или пуст. Ставки не применены, издержки занижены, вердикт оптимистичен."><td>${label}</td>
        <td class="num mono">н/д</td>
        <td class="share"><span class="costs-note">событий: ${events}</span></td></tr>`;
    }
    const share = (Math.abs(val) / absSum) * 100;
    const cls = val < 0 ? 'income' : '';
    const sign = val < 0 ? ' title="отрицательный funding — доход от финансирования"' : '';
    return `<tr${sign}><td>${label}</td>
      <td class="num mono ${cls}">${fmtPct(val / denom, 3)}</td>
      <td class="share"><div class="sbar"><i style="width:${share.toFixed(1)}%"></i></div></td></tr>`;
  }).join('');

  document.getElementById('costs-body').innerHTML = body +
    `<tr class="total"><td><b>Итого</b></td>
      <td class="num mono"><b>${fmtPct(sum / denom, 3)}</b></td>
      <td class="share"></td></tr>`;

  const note = document.getElementById('costs-note');
  if (note) {
    let text = 'Накопленные издержки за прогон, в долях начального капитала ' +
      '(equity₀ = ' + fmtNum(denom, 2) + '). Funding отрицательный — значит, ' +
      'финансирование приносило доход. Итог — чистая стоимость издержек.';
    if (!fundingAvailable) {
      text += ' Funding недоступен (файл отсутствует или пуст): ставки не ' +
        'применены, издержки занижены, вердикт оптимистичен.';
    } else if (isNum(extra.funding_events)) {
      text += ' Funding: событий ' + fmtInt(extra.funding_events) +
        ', привязано к барам ' + fmtInt(extra.funding_matched) + '.';
    }
    note.textContent = text;
  }
}

/* ── сборка страницы ────────────────────────────────────────── */

function renderApp(report) {
  const s = report.series;
  const extra = report.extra || {};
  const n = s.ts.length;
  const tsMs = parseTimestamps(s.ts);
  const sym = extra.symbol || '—';
  const tf = extra.timeframe ? ' · ' + extra.timeframe : '';

  const meta = document.getElementById('top-meta');
  if (meta) meta.textContent = `${sym}${tf} · баров: ${fmtInt(n)}`;

  const costs = extra.costs_total || {};
  const costsZero = !isNum(costs.fee) && !isNum(costs.slippage) && !isNum(costs.funding);

  document.getElementById('app').innerHTML = `
    ${capacityBanner(extra)}
    <div id="verdict-box"></div>
    <div class="card">
      <div class="card-head">
        <h2>Кривая доходности</h2>
        <div class="readout mono" id="ro-equity"></div>
      </div>
      <canvas id="cv-equity" role="img" aria-label="Кривая доходности (equity) за прогон" style="height:240px"></canvas>
      <div class="legend">
        <span><i class="ln" style="background:${C.sky}"></i>equity</span>
        <span><i class="ln dash"></i>стартовый капитал</span>
      </div>
      ${tsMs ? '' : '<div class="chart-note">В отчёте нет корректных меток времени — по оси номера баров.</div>'}
    </div>
    <div class="card">
      <div class="card-head">
        <h2>Просадка</h2>
        <div class="readout mono" id="ro-drawdown"></div>
      </div>
      <canvas id="cv-drawdown" role="img" aria-label="Просадка от пика equity" style="height:120px"></canvas>
    </div>
    <div class="card">
      <div class="card-head">
        <h2>Цена и позиция</h2>
        <div class="readout mono" id="ro-price"></div>
      </div>
      <canvas id="cv-price" role="img" aria-label="Цена закрытия с разметкой позиции long/short" style="height:240px"></canvas>
      <div class="legend">
        <span><i class="ln" style="background:${C.slate}"></i>close</span>
        ${Array.isArray(s.high) && Array.isArray(s.low)
          ? '<span><i class="bx" style="background:rgba(148,163,184,.25)"></i>high–low</span>' : ''}
        <span><i class="bx" style="background:rgba(34,197,94,.35);border:1px solid rgba(34,197,94,.5)"></i>long</span>
        <span><i class="bx" style="background:rgba(239,68,68,.35);border:1px solid rgba(239,68,68,.5)"></i>short</span>
      </div>
    </div>
    <div class="card">
      <div class="card-head"><h2>Издержки, доля начального капитала</h2></div>
      <table>
        <thead><tr><th>Статья</th><th class="num">Значение</th><th class="share">Структура</th></tr></thead>
        <tbody id="costs-body"></tbody>
      </table>
      <div class="costs-note" id="costs-note"></div>
      ${costsZero ? '<div class="costs-note">В extra.costs_total нет данных — издержки не переданы.</div>' : ''}
    </div>`;

  renderVerdict(report);
  renderCosts(report);

  /* readout-тексты */
  const eqReadout = () => {
    const last = (() => {
      for (let i = n - 1; i >= 0; i--) if (isNum(s.equity[i])) return s.equity[i];
      return null;
    })();
    const start = (() => {
      for (let i = 0; i < n; i++) if (isNum(s.equity[i])) return s.equity[i];
      return null;
    })();
    return `equity <b>${fmtNum(last, 4)}</b> · от старта ${fmtPct(isNum(last) && isNum(start) ? last / start - 1 : null, 1)}`;
  };
  const eqHover = i => `бар ${fmtInt(i)}${tsMs && tsMs[i] !== null ? ' · ' + fmtDateTime(tsMs[i]) : ''} · ` +
    `equity <b>${fmtNum(s.equity[i], 4)}</b> · просадка ${fmtPct(s.drawdown ? s.drawdown[i] : null, 1)}`;

  const ddMin = (() => {
    let m = null;
    if (!Array.isArray(s.drawdown)) return null;
    for (const v of s.drawdown) if (isNum(v) && (m === null || v < m)) m = v;
    return m;
  })();
  const ddHover = i => `бар ${fmtInt(i)}${tsMs && tsMs[i] !== null ? ' · ' + fmtDateTime(tsMs[i]) : ''} · ` +
    `просадка <b>${fmtPct(s.drawdown ? s.drawdown[i] : null, 1)}</b>`;

  const posShare = (() => {
    if (!Array.isArray(s.position)) return null;
    let long = 0, short = 0, live = 0;
    for (const v of s.position) {
      if (!isNum(v)) continue;
      live++;
      if (v > 0) long++; else if (v < 0) short++;
    }
    if (!live) return null;
    return { long: long / live, short: short / live };
  })();
  const priceLast = (() => {
    for (let i = n - 1; i >= 0; i--) if (isNum(s.close[i])) return s.close[i];
    return null;
  })();
  const priceDefault = posShare
    ? `close <b>${fmtPrice(priceLast)}</b> · long ${fmtPct(posShare.long, 0)} времени · short ${fmtPct(posShare.short, 0)}`
    : `close <b>${fmtPrice(priceLast)}</b>`;
  const posName = v => (v > 0 ? 'long' : v < 0 ? 'short' : 'вне рынка');
  const priceHover = i => `бар ${fmtInt(i)}${tsMs && tsMs[i] !== null ? ' · ' + fmtDateTime(tsMs[i]) : ''} · ` +
    `close <b>${fmtPrice(s.close[i])}</b>` +
    (Array.isArray(s.position) ? ` · ${posName(s.position[i])}` : '');

  registerChart(document.getElementById('cv-equity'), n, document.getElementById('ro-equity'),
    eqReadout(), eqHover, hover => drawEquityChart(document.getElementById('cv-equity'), s.equity, n, tsMs, hover));
  registerChart(document.getElementById('cv-drawdown'), n, document.getElementById('ro-drawdown'),
    `макс. просадка <b>${fmtPct(ddMin, 1)}</b>`, ddHover,
    hover => drawDrawdownChart(document.getElementById('cv-drawdown'), s.drawdown || [], n, tsMs, hover));
  registerChart(document.getElementById('cv-price'), n, document.getElementById('ro-price'),
    priceDefault, priceHover,
    hover => drawPriceChart(document.getElementById('cv-price'), s, n, tsMs, hover));

  redrawAll();

  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(redrawAll, 120);
  });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
}

/* ── вход ───────────────────────────────────────────────────── */

function init() {
  const report = window.ALPHA_REPORT;

  if (!report) {
    renderError('Отчёт не найден', `
      <p>Файл <code>report.js</code> не загрузился или содержит
         <code>window.ALPHA_REPORT = null</code> — дашборду нечего показывать.</p>
      <p>Сгенерируйте отчёт и положите его рядом с <code>index.html</code>:</p>
      <ul>
        <li><code>uv run alpha-lab validate --config configs/experiments/mr_base.yaml --out dashboard</code></li>
        <li>либо скопируйте готовый <code>report.js</code> из <code>reports/&lt;experiment_id&gt;/</code></li>
      </ul>
      <p>Данные подключаются как <code>&lt;script src="report.js"&gt;</code>, потому что
         <code>fetch()</code> к файлам на <code>file://</code> блокируется CORS.</p>`);
    return;
  }

  const major = majorOf(report.schema_version);
  if (major !== REQUIRED_MAJOR) {
    const shown = report.schema_version === undefined ? 'не указана' : `«${report.schema_version}»`;
    renderError('Несовместимая версия схемы', `
      <p>Версия схемы отчёта: <code>${esc(shown)}</code>, дашборд понимает
         <code>${REQUIRED_MAJOR}.x</code>.</p>
      <p>Пустые графики в этой ситуации были бы хуже отказа: контракт мог измениться,
         и нарисованное выглядело бы правдоподобно, но было бы неверным.</p>
      <p>Обновите дашборд или перегенерируйте отчёт совместимой версией
         <code>alpha-lab validate</code>.</p>`);
    return;
  }

  const problem = structuralProblem(report);
  if (problem) {
    renderError('Повреждённый отчёт', `<p>${problem}</p>`);
    return;
  }

  renderApp(report);
}

document.addEventListener('DOMContentLoaded', init);
