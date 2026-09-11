'use strict';

/* Alpha Lab — вид воронки отбора (P4) поверх статического отчёта.
 *
 * Данные приходят из funnel.js как window.ALPHA_FUNNEL — тот же контракт, что
 * у report.js: file:// блокирует fetch(), поэтому <script src>. Если воронки
 * нет (файл не сгенерирован), представление молча не появляется: отсутствие
 * артефакта — не ошибка страницы.
 *
 * Контракт: schema_version, thresholds, funnel, screening, by_timeframe,
 * by_symbol, survivors, errors (см. src/alpha_lab/funnel.py). Несовместимая
 * мажорная версия — видимое сообщение вместо правдоподобных, но неверных чисел.
 */

(function () {
  const REQUIRED_MAJOR = 1;
  const funnel = window.ALPHA_FUNNEL;
  if (!funnel) return;

  const app = document.getElementById('app');
  if (!app) return;

  function majorOf(version) {
    const m = /^\s*(\d+)\s*\./.exec(String(version));
    return m ? parseInt(m[1], 10) : NaN;
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function num(v, digits) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return Number(v).toLocaleString('ru-RU', {
      minimumFractionDigits: digits, maximumFractionDigits: digits
    });
  }

  function pct(f) {
    if (f === null || f === undefined || !isFinite(f)) return '—';
    return (f * 100).toFixed(1).replace('.', ',') + '%';
  }

  function int(v) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    return Math.round(v).toLocaleString('ru-RU');
  }

  const card = document.createElement('section');
  card.className = 'card';
  card.id = 'funnel-card';

  if (majorOf(funnel.schema_version) !== REQUIRED_MAJOR) {
    card.innerHTML =
      `<div class="alert warn"><h3>Воронка: несовместимая версия схемы</h3>
       <p>funnel.js содержит schema_version «${esc(funnel.schema_version)}», ` +
      `дашборд умеет ${REQUIRED_MAJOR}.x. Числа не показаны: ` +
      `частично совместимая воронка выглядела бы правдоподобно, но была бы неверна.</p></div>`;
    app.appendChild(card);
    return;
  }

  function stepRows(block) {
    const total = block.n || 0;
    return (block.steps || []).map(step => {
      const width = Math.max(0, Math.min(100, (step.fraction_of_previous || 0) * 100));
      return `<tr>
        <td>${esc(step.label)}</td>
        <td class="num mono">${int(step.n)}</td>
        <td class="num mono">${pct(step.fraction_of_total)}</td>
        <td class="num mono">${pct(step.fraction_of_previous)}</td>
        <td style="width:180px"><div style="height:6px;border-radius:3px;background:rgba(255,255,255,.08)">
          <div style="height:6px;border-radius:3px;width:${width}%;background:#38bdf8"></div></div></td>
      </tr>`;
    }).join('') + (block.n === 0 ? '' :
      `<tr><td><b>alive</b></td>
        <td class="num mono"><b>${int(block.alive.n)}</b></td>
        <td class="num mono">${pct(block.alive.fraction_of_total)}</td>
        <td class="num mono">${pct(block.alive.fraction_of_previous)}</td>
        <td></td></tr>`) + `<tr><td colspan="5" style="color:#64748b;font-size:11.5px">когорта: ${int(total)} вердиктов (строки-ошибки исключены)</td></tr>`;
  }

  function screeningHtml(block) {
    if (!block) return '';
    const candidates = block.candidates || 0;
    return `<div class="alert warn" style="margin-top:14px">
      <h3>Черновая воронка (screening): alive не вынесен</h3>
      <p>${esc(block.note)}</p>
      <p>Кандидатов, прошедших все гейты: <b>${int(candidates)}</b> из ${int(block.n)}.
      Минимальный достижимый p-value: <span class="mono">1/${int(block.n_permutations) + 1} ≈ ${num(block.min_achievable_p, 4)}</span> —
      черновой вердикт может только отсеять, но не принять.</p></div>`;
  }

  function breakdownTable(title, rows, keyName) {
    if (!rows || !rows.length) return '';
    return `<h3 style="margin:16px 0 6px;font-size:13px">${esc(title)}</h3>
      <table><thead><tr><th>${esc(keyName)}</th><th class="num">вердиктов</th>
      <th class="num">alive</th><th class="num">черновых</th><th class="num">ошибок</th></tr></thead>
      <tbody>${rows.map(g => `<tr>
        <td class="mono">${esc(g[keyName])}</td>
        <td class="num mono">${int(g.n)}</td>
        <td class="num mono">${int(g.alive)}</td>
        <td class="num mono">${int(g.screening)}</td>
        <td class="num mono">${int(g.errors)}</td></tr>`).join('')}</tbody></table>`;
  }

  const survivors = funnel.survivors || { n: 0, top: [], note: '' };
  let survivorsHtml;
  if (survivors.n > 0) {
    const head = `<th>#</th><th>config_id</th><th>символ</th><th>тайм</th>
      <th class="num">Sharpe</th><th class="num">DSR</th><th class="num">p-value</th>
      <th class="num">PBO</th><th class="num">сделок</th><th class="num">max DD</th>
      <th class="num">издержки</th>`;
    const rows = survivors.top.map((s, i) => `<tr>
      <td class="num mono">${i + 1}</td>
      <td class="mono">${esc(s.config_id)}</td>
      <td class="mono">${esc(s.symbol)}</td>
      <td class="mono">${esc(s.timeframe)}</td>
      <td class="num mono">${num(s.sharpe, 2)}</td>
      <td class="num mono">${num(s.dsr, 4)}</td>
      <td class="num mono">${num(s.p_value, 4)}</td>
      <td class="num mono">${num(s.pbo, 2)}</td>
      <td class="num mono">${int(s.trades)}</td>
      <td class="num mono">${pct(s.max_dd)}</td>
      <td class="num mono">${num(s.cost_total, 4)}</td></tr>`).join('');
    survivorsHtml = `<h3 style="margin:16px 0 6px;font-size:13px">Топ выживших</h3>
      <p style="color:#94a3b8;font-size:12px">${esc(survivors.note)}</p>
      <table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
  } else {
    survivorsHtml = `<div class="alert warn" style="margin-top:14px">
      <h3>Выживших нет: 0</h3><p>${esc(survivors.note)}</p></div>`;
  }

  const warnings = (funnel.warnings || []).map(w =>
    `<p>${esc(w)}</p>`).join('');
  const errors = funnel.errors || { n: 0, messages: [] };
  const errorsHtml = errors.n
    ? `<div class="alert warn" style="margin-top:14px">
         <h3>Ошибки: ${int(errors.n)} — не гипотезы, из воронки исключены</h3>
         ${errors.messages.map(e => `<p class="mono">${esc(e.symbol)} ${esc(e.timeframe)} ${esc(e.config_id)}: ${esc(e.error)}</p>`).join('')}
       </div>`
    : '';

  card.innerHTML = `
    <div class="card-head"><h2>Воронка отбора</h2>
      <div class="readout mono">${int(funnel.n_configs)} конфигураций · ${
        funnel.grade === 'mixed' ? 'смешанный грейд' :
        funnel.grade === 'screening' ? 'черновой прогон' : 'полные вердикты'}</div>
    </div>
    ${warnings ? `<div class="alert warn"><h3>Оговорки</h3>${warnings}</div>` : ''}
    <h3 style="margin:12px 0 6px;font-size:13px">Полная воронка</h3>
    <table><thead><tr><th>гейт</th><th class="num">прошло</th>
      <th class="num">доля от когорты</th><th class="num">доля от предыдущего</th>
      <th></th></tr></thead><tbody>${stepRows(funnel.funnel)}</tbody></table>
    ${screeningHtml(funnel.screening)}
    ${breakdownTable('Разрез по таймфреймам', funnel.by_timeframe, 'timeframe')}
    ${breakdownTable('Разрез по символам', funnel.by_symbol, 'symbol')}
    ${survivorsHtml}
    ${errorsHtml}
    <p style="color:#64748b;font-size:11.5px;margin-top:14px">Воронка построена по
    хранилищу ${esc((funnel.source && funnel.source.store) || '—')} ·
    схема ${esc(funnel.schema_version)}</p>`;

  app.appendChild(card);
})();
