'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   Отрисовка превью модели.

   Рисует НЕ Pine (его нельзя исполнить в браузере), а выход проверенной
   Python-модели, выгруженный scripts/export_preview.py. Смысл в том, что
   модель — эталон: Pine-порт сверен с ней побитово, значит картинка здесь
   показывает ровно то, что индикатор обязан нарисовать в TradingView.
   Расхождение с этой картинкой — дефект индикатора, а не превью.
   ═══════════════════════════════════════════════════════════════════════════ */

const C = {
  bg: '#0b0e14', panel: '#0d1117', grid: 'rgba(255,255,255,.05)',
  text: '#cbd5e1', dim: '#64748b',
  up: '#22c55e', down: '#ef4444',
  mean: '#f59e0b', band: 'rgba(245,158,11,.16)',
  z: '#38bdf8', zline: 'rgba(148,163,184,.35)',
  entryL: '#2dd4bf', entryS: '#fb7185',
  stop: '#ef4444', take: '#22c55e', time: '#94a3b8',
};

function fitCanvas(cv, h) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  cv.width = w * dpr;
  cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/** Основная панель: цена, среднее, полосы, сделки. */
function drawPrice(cv, d, view) {
  const H = 300;
  const { ctx, w, h } = fitCanvas(cv, H);
  const padL = 8, padR = 62, padT = 10, padB = 22;

  const i0 = view.i0, i1 = view.i1;
  const bars = d.bars.slice(i0, i1 + 1);
  if (!bars.length) return;

  let lo = Infinity, hi = -Infinity;
  for (const b of bars) { lo = Math.min(lo, b.l); hi = Math.max(hi, b.h); }
  const seriesFrom = d.series.mean.slice(i0, i1 + 1);
  const upFrom = d.series.upper.slice(i0, i1 + 1);
  const loFrom = d.series.lower.slice(i0, i1 + 1);
  for (const v of upFrom) if (v != null) hi = Math.max(hi, v);
  for (const v of loFrom) if (v != null) lo = Math.min(lo, v);
  const pad = (hi - lo) * 0.06 || 1;
  lo -= pad; hi += pad;

  const n = bars.length;
  const X = i => padL + (i / Math.max(n - 1, 1)) * (w - padL - padR);
  const Y = v => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);

  // Сетка и подписи цены
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  ctx.fillStyle = C.dim; ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  for (let g = 0; g <= 4; g++) {
    const v = lo + (hi - lo) * g / 4;
    const y = Math.round(Y(v)) + .5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(v > 1000 ? 0 : 2), w - padR + 6, y);
  }

  // Полоса канала
  ctx.beginPath();
  let started = false;
  upFrom.forEach((v, i) => {
    if (v == null) return;
    if (!started) { ctx.moveTo(X(i), Y(v)); started = true; } else ctx.lineTo(X(i), Y(v));
  });
  for (let i = loFrom.length - 1; i >= 0; i--) {
    const v = loFrom[i];
    if (v != null) ctx.lineTo(X(i), Y(v));
  }
  ctx.closePath(); ctx.fillStyle = C.band; ctx.fill();

  // Бары: тонкие свечи
  const bw = Math.max(1, (w - padL - padR) / n * 0.7);
  bars.forEach((b, i) => {
    const up = b.c >= b.o;
    ctx.strokeStyle = up ? C.up : C.down;
    ctx.fillStyle = up ? C.up : C.down;
    ctx.lineWidth = 1;
    const x = X(i);
    ctx.beginPath(); ctx.moveTo(x, Y(b.h)); ctx.lineTo(x, Y(b.l)); ctx.stroke();
    const yo = Y(b.o), yc = Y(b.c);
    ctx.fillRect(x - bw / 2, Math.min(yo, yc), bw, Math.max(1, Math.abs(yc - yo)));
  });

  // Среднее
  ctx.beginPath(); started = false;
  seriesFrom.forEach((v, i) => {
    if (v == null) return;
    if (!started) { ctx.moveTo(X(i), Y(v)); started = true; } else ctx.lineTo(X(i), Y(v));
  });
  ctx.strokeStyle = C.mean; ctx.lineWidth = 1.5; ctx.stroke();

  // Сделки: уровни стопа и тейка, метки входа и выхода
  for (const t of d.trades) {
    if (t.exit_i < i0 || t.entry_i > i1) continue;
    const a = Math.max(t.entry_i, i0) - i0;
    const b = Math.min(t.exit_i, i1) - i0;
    const xa = X(a), xb = X(b);

    // Зона удержания
    ctx.fillStyle = t.dir > 0 ? 'rgba(45,212,191,.06)' : 'rgba(251,113,133,.06)';
    ctx.fillRect(xa, padT, Math.max(xb - xa, 1), h - padT - padB);

    if (t.sl != null) {
      ctx.strokeStyle = 'rgba(239,68,68,.45)'; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xa, Y(t.sl)); ctx.lineTo(xb, Y(t.sl)); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (t.tp != null) {
      ctx.strokeStyle = 'rgba(34,197,94,.45)'; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(xa, Y(t.tp)); ctx.lineTo(xb, Y(t.tp)); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Вход
    const yEntry = Y(t.entry_p);
    ctx.fillStyle = t.dir > 0 ? C.entryL : C.entryS;
    ctx.beginPath();
    const s = 4;
    if (t.dir > 0) { ctx.moveTo(xa, yEntry + s); ctx.lineTo(xa - s, yEntry - s); ctx.lineTo(xa + s, yEntry - s); }
    else { ctx.moveTo(xa, yEntry - s); ctx.lineTo(xa - s, yEntry + s); ctx.lineTo(xa + s, yEntry + s); }
    ctx.closePath(); ctx.fill();

    // Выход — цветом причины
    const yExit = Y(t.exit_p);
    ctx.strokeStyle = t.kind === 'стоп' ? C.stop : t.kind === 'тейк' ? C.take : C.time;
    ctx.lineWidth = 1.8;
    const k = 3.5;
    ctx.beginPath();
    ctx.moveTo(xb - k, yExit - k); ctx.lineTo(xb + k, yExit + k);
    ctx.moveTo(xb + k, yExit - k); ctx.lineTo(xb - k, yExit + k);
    ctx.stroke();
  }

  // Подписи времени
  ctx.fillStyle = C.dim; ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const ticks = 5;
  for (let g = 0; g <= ticks; g++) {
    const i = Math.round(i0 + (i1 - i0) * g / ticks);
    const t = d.bars[i] && d.bars[i].t;
    if (!t) continue;
    const lbl = t.slice(0, 10);
    const x = X(Math.min(i, i1) - i0);
    ctx.fillText(lbl, Math.max(30, Math.min(w - padR - 30, x)), h - padB + 5);
  }
}

/** Нижняя панель: z-скор и пороги входа. */
function drawZ(cv, d, view, kEntry) {
  const H = 110;
  const { ctx, w, h } = fitCanvas(cv, H);
  const padL = 8, padR = 62, padT = 8, padB = 14;

  const i0 = view.i0, i1 = view.i1;
  const zs = d.series.z.slice(i0, i1 + 1);
  if (!zs.length) return;

  const lim = Math.max(kEntry * 1.6, Math.max(...zs.map(v => Math.abs(v ?? 0))) * 1.05);
  const X = i => padL + (i / Math.max(zs.length - 1, 1)) * (w - padL - padR);
  const Y = v => padT + (1 - (v + lim) / (2 * lim)) * (h - padT - padB);

  // Нулевая линия и пороги
  ctx.strokeStyle = C.zline; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(w - padR, Y(0)); ctx.stroke();

  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(239,68,68,.5)';
  ctx.beginPath(); ctx.moveTo(padL, Y(kEntry)); ctx.lineTo(w - padR, Y(kEntry)); ctx.stroke();
  ctx.strokeStyle = 'rgba(45,212,191,.5)';
  ctx.beginPath(); ctx.moveTo(padL, Y(-kEntry)); ctx.lineTo(w - padR, Y(-kEntry)); ctx.stroke();
  ctx.setLineDash([]);

  // Кривая z
  ctx.beginPath();
  zs.forEach((v, i) => {
    if (v == null) return;
    if (i === 0) ctx.moveTo(X(i), Y(v)); else ctx.lineTo(X(i), Y(v));
  });
  ctx.strokeStyle = C.z; ctx.lineWidth = 1.4; ctx.stroke();

  ctx.fillStyle = C.dim; ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  ctx.fillText('+' + kEntry.toFixed(1), w - padR + 6, Y(kEntry));
  ctx.fillText('0', w - padR + 6, Y(0));
  ctx.fillText('-' + kEntry.toFixed(1), w - padR + 6, Y(-kEntry));
}

window.AlphaChart = { drawPrice, drawZ, C };
