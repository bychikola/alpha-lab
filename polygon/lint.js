'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   Линтер Pine Script.

   Зачем он нужен. Компилятор TradingView — серверный: он отдаёт код ошибки
   и, если повезёт, номер строки, но не говорит, ЧТО именно не так. Две
   ошибки, на которые натыкаются чаще всего, выглядят так:

       CE10016  Extra closing parenthesis
       CE10017  Missing enclosing character in the literal string

   Обе — синтаксические, и обе ловятся локально. Хуже того, они часто
   ПРИХОДЯТ ОТ ОДНОЙ ПРИЧИНЫ: разорванное переносом строковое выражение.
   Pine читает разорванный литерал как незакрытый (CE10017), а весь код
   после него — как содержимое строки, поэтому какая-нибудь закрывающая
   скобка оказывается «лишней» (CE10016). Именно так и было в этом проекте:
   ошибка указывала на строку 64, а причина искалась в скобках.

   Линтер проверяет то, что можно проверить без компилятора. Он НЕ заменяет
   компилятор: типы, версии функций и всё, что требует семантики, он не
   знает. Но ошибки разбора — те самые, что съедают вечер, — находит.
   ═══════════════════════════════════════════════════════════════════════════ */

const RULES = [];

/** Регистрирует правило. */
function rule(id, title, check) {
  RULES.push({ id, title, check });
}

/* ── Разбор строки: вырезает комментарий и помечает литералы ──────────────
   Возвращает { code, literals, unterminated }:
     code         — строка без комментария и без содержимого литералов,
                    с пробелами на месте вырезанного (длины сохраняются,
                    чтобы столбцы в сообщениях совпадали с исходником);
     literals     — массив { start, end } границ литералов;
     unterminated — true, если кавычка не закрылась до конца строки.
   Апостиль в Pine — тоже разделитель строк, поэтому учитываются оба.      */
function scan(line) {
  let code = '';
  const literals = [];
  let i = 0, unterminated = false;
  const n = line.length;

  while (i < n) {
    const ch = line[i];

    if (ch === '/' && line[i + 1] === '/') break;      // комментарий до конца

    if (ch === '"' || ch === "'") {
      const start = i;
      i++;
      while (i < n && line[i] !== ch) {
        if (line[i] === '\\') i++;                     // экранирование
        i++;
      }
      if (i >= n) {
        unterminated = true;
        code += ' '.repeat(n - start);
        break;
      }
      i++;                                             // закрывающая кавычка
      literals.push({ start, end: i });
      code += ' '.repeat(i - start);
      continue;
    }

    code += ch;
    i++;
  }
  return { code, literals, unterminated };
}

/** Убирает из строки содержимое кавычек — для проверок имён и функций. */
function stripLiterals(line) {
  return scan(line).code;
}

/* ═══════════════════════════ ПРАВИЛА ═══════════════════════════════════ */

rule('CE10017', 'Незакрытая строка', (line, ctx) => {
  const s = scan(line);
  if (!s.unterminated) return null;
  return {
    severity: 'error',
    message: 'Строковый литерал не закрыт на этой строке.',
    hint: 'Pine не позволяет строке переноситься. Закройте кавычку здесь же — ' +
          'либо перенесите весь текст на одну строку, либо разбейте выражение ' +
          'на несколько переменных.',
  };
});

rule('CE10017+', 'Перенос строкового выражения', (line, ctx) => {
  const s = scan(line);
  if (!s.literals.length) return null;
  // Строка кончается оператором, а за ним — перенос.
  if (!/[+\-*/?:,]\s*$/.test(line)) return null;
  return {
    severity: 'error',
    message: 'Выражение со строкой разорвано переносом.',
    hint: 'Pine читает перенесённый литерал как незакрытый. Запишите всё ' +
          'выражение одной строкой — длинные строки компилятор принимает.',
  };
});

rule('CE10016', 'Лишняя закрывающая скобка', (line, ctx) => {
  const s = scan(line);
  for (const ch of s.code) {
    if (ch === '(') ctx.depth++;
    else if (ch === ')') {
      ctx.depth--;
      if (ctx.depth < 0) {
        ctx.depth = 0;
        return {
          severity: 'error',
          message: 'Закрывающих скобок больше, чем открывающих.',
          hint: 'Проверьте выражение целиком. Если скобки на вид сходятся, ' +
                'ищите выше разорванную строку: из-за неё код ниже считается ' +
                'содержимым литерала, и скобка «лишняя» только для парсера.',
        };
      }
    }
  }
  return null;
});

rule('перенос', 'Выражение разорвано переносом', (line, ctx) => {
  if (ctx.depth > 0) return null;                 // внутри скобок перенос можно
  const s = scan(line);
  if (s.literals.length) return null;             // это правило CE10017+
  if (!/[+\-*/?:]\s*$/.test(line)) return null;
  if (/^\s*\//.test(line)) return null;           // комментарий
  return {
    severity: 'warn',
    message: 'Строка заканчивается оператором — Pine должен продолжить выражение.',
    hint: 'Pine умеет переносить по оператору, но разбор такого кода хрупкий. ' +
          'Надёжнее записать выражение одной строкой.',
  };
});

rule('if(...)', 'Скобки вокруг условия', (line) => {
  const s = stripLiterals(line);
  if (/\bif\s*\(/.test(s) || /\bwhile\s*\(/.test(s)) {
    return {
      severity: 'error',
      message: 'Pine не принимает скобки вокруг условия if/while.',
      hint: 'Пишите  if условие  без скобок.',
    };
  }
  return null;
});

rule('version', 'Отсутствует директива версии', (line, ctx) => {
  if (ctx.lineNo !== 1) return null;
  if (!/^\/\/@version=/.test(line)) {
    return {
      severity: 'error',
      message: 'Первая строка должна содержать //@version=6.',
      hint: 'Без директивы компилятор не знает версию языка.',
    };
  }
  return null;
});

rule('табы', 'Табуляция в отступе', (line) => {
  if (/^\t+/.test(line) || /^ *\t/.test(line)) {
    return {
      severity: 'error',
      message: 'В отступе табуляция.',
      hint: 'Pine требует отступы пробелами, иначе блоки вложенности рвутся.',
    };
  }
  return null;
});

rule(':=', 'Присваивание необъявленной переменной', (line, ctx) => {
  const s = stripLiterals(line);
  const m = s.match(/^\s*([A-Za-z_]\w*)\s*:=/);
  if (!m) return null;
  if (!ctx.declared.has(m[1])) {
    return {
      severity: 'error',
      message: `Переменная «${m[1]}» не объявлена до присваивания через :=`,
      hint: 'Сначала объявите: var ' + m[1] + ' = ... — иначе Pine не знает тип.',
    };
  }
  return null;
});

rule('резерв', 'Зарезервированное слово как имя', (line) => {
  const RESERVED = ['if', 'else', 'for', 'while', 'switch', 'var', 'varip',
    'true', 'false', 'na', 'and', 'or', 'not', 'series', 'simple', 'const',
    'input', 'int', 'float', 'bool', 'string', 'color', 'line', 'label', 'table'];
  const s = stripLiterals(line);
  const m = s.match(/^\s*(?:var\s+\w+\s+)?([A-Za-z_]\w*)\s*=\s*(?!=)/);
  if (m && RESERVED.includes(m[1])) {
    return {
      severity: 'error',
      message: `«${m[1]}» — зарезервированное слово, его нельзя использовать как имя.`,
      hint: 'Переименуйте переменную.',
    };
  }
  return null;
});

rule('функция', 'Неизвестная функция Pine', (line) => {
  const KNOWN = new Set([
    'ta.sma', 'ta.ema', 'ta.wma', 'ta.vwma', 'ta.rma', 'ta.hma', 'ta.atr',
    'ta.tr', 'ta.stdev', 'ta.variance', 'ta.rsi', 'ta.adx', 'ta.dmi', 'ta.correlation',
    'ta.crossover', 'ta.crossunder', 'ta.highest', 'ta.lowest', 'ta.barssince',
    'ta.change', 'ta.cum', 'ta.mom', 'ta.roc', 'ta.percentrank', 'ta.median',
    'math.abs', 'math.max', 'math.min', 'math.round', 'math.floor', 'math.ceil',
    'math.sqrt', 'math.pow', 'math.log', 'math.exp', 'math.sign', 'math.sum',
    'math.avg', 'math.todegrees', 'math.toradians', 'math.random', 'math.pi',
    'str.tostring', 'str.tonumber', 'str.length', 'str.contains', 'str.replace',
    'str.split', 'str.substring', 'str.format', 'str.upper', 'str.lower',
    'input.int', 'input.float', 'input.bool', 'input.string', 'input.color',
    'input.timeframe', 'input.symbol', 'input.session', 'input.source', 'input.price',
    'array.new_int', 'array.new_float', 'array.new_bool', 'array.new_string',
    'array.new_color', 'array.new_line', 'array.new_label', 'array.new_box',
    'array.new_table', 'array.push', 'array.pop', 'array.get', 'array.set',
    'array.size', 'array.clear', 'array.shift', 'array.unshift', 'array.remove',
    'array.insert', 'array.slice', 'array.reverse', 'array.sort', 'array.sum',
    'array.avg', 'array.min', 'array.max', 'array.stdev', 'array.covariance',
    'array.from', 'array.includes', 'array.indexof', 'array.lastindexof',
    'table.new', 'table.cell', 'table.clear', 'table.set_bgcolor', 'table.merge_cells',
    'line.new', 'line.set_x2', 'line.set_y2', 'line.set_xy1', 'line.set_xy2',
    'line.set_color', 'line.set_width', 'line.set_style', 'line.delete',
    'label.new', 'label.set_x', 'label.set_y', 'label.set_text', 'label.delete',
    'box.new', 'box.set_right', 'box.delete',
    'plot', 'plotshape', 'plotchar', 'plotcandle', 'plotbar', 'plotarrow',
    'hline', 'fill', 'bgcolor', 'barcolor', 'alert', 'alertcondition',
    'request.security', 'request.security_lower_tf', 'request.financial',
    'strategy.entry', 'strategy.exit', 'strategy.close', 'strategy.order',
    'strategy.position_size', 'strategy.equity', 'strategy.opentrades',
    'color.new', 'color.rgb', 'color.from_gradient',
    'timeframe.period', 'timeframe.multiplier', 'timeframe.isintraday',
    'syminfo.ticker', 'syminfo.mintick', 'syminfo.type',
    'barstate.islast', 'barstate.isfirst', 'barstate.isnew', 'barstate.isconfirmed',
    'display.data_window', 'display.all', 'display.none',
    'shape.triangleup', 'shape.triangledown', 'shape.xcross', 'shape.circle',
    'location.abovebar', 'location.belowbar', 'location.absolute', 'location.top',
    'size.tiny', 'size.small', 'size.normal', 'size.large', 'size.huge', 'size.auto',
    'format.mintick', 'format.percent', 'format.volume', 'format.price',
    'position.top_right', 'position.bottom_right', 'position.top_left',
    'position.bottom_left', 'position.middle_center',
    'line.style_solid', 'line.style_dashed', 'line.style_dotted',
  ]);
  const s = stripLiterals(line);
  const re = /\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\s*\(/g;
  let m;
  while ((m = re.exec(s))) {
    const full = m[1] + '.' + m[2];
    if (!KNOWN.has(full)) {
      return {
        severity: 'warn',
        message: `Неизвестная функция «${full}»`,
        hint: 'Возможно опечатка — или функция появилась в другой версии Pine. ' +
              'Компилятор проверит точно, это лишь предупреждение.',
      };
    }
  }
  return null;
});

rule('отступ', 'Неровный отступ блока', (line, ctx) => {
  const m = line.match(/^( +)/);
  if (!m) return null;
  const n = m[1].length;
  if (n % 4 !== 0) {
    return {
      severity: 'warn',
      message: `Отступ ${n} пробелов не кратен 4.`,
      hint: 'Pine не требует именно четырёх, но ровный шаг избавляет от ' +
            'загадочных ошибок вложенности.',
    };
  }
  return null;
});

/* ═══════════════════════════ ЗАПУСК ═══════════════════════════════════ */

function lint(source) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  const ctx = { depth: 0, declared: new Set(), lineNo: 0 };
  const out = [];

  lines.forEach((line, idx) => {
    ctx.lineNo = idx + 1;

    for (const r of RULES) {
      const res = r.check(line, ctx);
      if (res) {
        out.push({
          line: idx + 1,
          col: 1,
          rule: r.id,
          title: r.title,
          severity: res.severity,
          message: res.message,
          hint: res.hint || '',
          text: line,
        });
      }
    }

    // Пополняем таблицу объявленных имён — нужно правилу `:=`.
    const s = stripLiterals(line);
    const m = s.match(/^\s*(?:var(?:ip)?\s+)?(?:[A-Za-z_]\w*(?:<[^>]*>)?\s+)?([A-Za-z_]\w*)\s*=(?!=)/);
    if (m) ctx.declared.add(m[1]);
  });

  if (ctx.depth !== 0) {
    out.push({
      line: lines.length,
      col: 1,
      rule: 'скобки',
      title: 'Незакрытая скобка',
      severity: 'error',
      message: ctx.depth > 0
        ? `Открыто скобок больше, чем закрыто, на ${ctx.depth}.`
        : 'Закрыто скобок больше, чем открыто.',
      hint: 'Проверьте многострочные вызовы: каждая открытая скобка должна ' +
            'закрыться.',
      text: '',
    });
  }

  return {
    issues: out.sort((a, b) => a.line - b.line),
    lines: lines.length,
  };
}

window.PineLint = { lint, RULES };
