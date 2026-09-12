"""Поиск символов, ломающих лексер Pine.

Pine считает строковым литералом всё, что открыто `"` ИЛИ `'`. Один случайный
апостроф в комментарии открывает строку, которую никто не закрывает, — и
дальше компилятор сыпет ошибками в местах, к причине не относящихся:
CE10016 «лишняя закрывающая скобка» (скобки внутри «строки» стали литералами)
и CE10017 «строка не закрыта».

Скрипт печатает:
  * строки с нечётным числом `"` или `'`;
  * необычные типографские кавычки, которые могли попасть при копировании
    (они ВЫГЛЯДЯТ как кавычки, но Pine их не распознаёт);
  * неразрывные пробелы и прочую невидимую типографику.

Запуск:
    uv run python scripts/check_pine_lex.py pine/mr_quant_v2.pine
"""
from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Символы, которые выглядят как кавычки, но разделителями строк в Pine не
# являются. Попадают при копировании из чата, документа или веб-страницы.
LOOKALIKE_QUOTES = {
    "‘": "‘ левая одиночная",
    "’": "’ правая одиночная (типографский апостроф)",
    "‚": "‚ нижняя одиночная",
    "‛": "‛ одиночная верхняя",
    "“": "“ левая двойная",
    "”": "” правая двойная",
    "„": "„ нижняя двойная",
    "′": "′ штрих",
    "″": "″ двойной штрих",
    "«": "« кавычка-ёлочка",
    "»": "» кавычка-ёлочка",
    "`": "` обратный апостроф",
}

INVISIBLE = {
    " ": "неразрывный пробел",
    " ": "узкий неразрывный пробел",
    " ": "тонкий пробел",
    "​": "нулевой ширины",
    "﻿": "BOM",
    " ": "разделитель строк",
    " ": "разделитель абзацев",
}


def main(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    problems = 0

    print("① строки с нечётным числом разделителей строк:")
    for n, line in enumerate(lines, 1):
        dq = line.count('"')
        # Апостиль в Pine — тоже разделитель, но внутри слова он допустим
        # реже; считаем все.
        sq = line.count("'")
        if dq % 2 or sq % 2:
            print(f"   СТРОКА {n}: \" {dq} шт, ' {sq} шт")
            print(f"     {line[:100]}")
            problems += 1
    if not problems:
        print("   чисто")

    print("\n② кавычки-двойники (Pine их НЕ распознаёт как разделители):")
    found = 0
    for n, line in enumerate(lines, 1):
        hits = [name for ch, name in LOOKALIKE_QUOTES.items() if ch in line]
        if hits:
            print(f"   СТРОКА {n}: {', '.join(hits)}")
            print(f"     {line[:100]}")
            found += 1
    if not found:
        print("   чисто")
    problems += found

    print("\n③ невидимые символы:")
    inv = 0
    for n, line in enumerate(lines, 1):
        hits = [name for ch, name in INVISIBLE.items() if ch in line]
        if hits:
            print(f"   СТРОКА {n}: {', '.join(hits)}")
            inv += 1
    if not inv:
        print("   чисто")
    problems += inv

    print("\n④ чем файл отличается от ASCII:")
    c = Counter(ch for ch in text if ord(ch) > 127)
    for ch, cnt in c.most_common():
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "?"
        print(f"   {ch!r:>8}  U+{ord(ch):04X}  ×{cnt:<4} {name}")

    print(f"\nподозрительных мест: {problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pine/mr_quant_v2.pine"
    raise SystemExit(main(target))
