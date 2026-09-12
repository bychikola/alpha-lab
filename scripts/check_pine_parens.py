"""Поиск лишней закрывающей скобки в Pine-скрипте.

Pine сообщает CE10016 («extra closing parenthesis») без номера строки, а
глазами такую ошибку в 440 строках искать долго. Скрипт считает глубину
вложенности по строкам, вырезая строковые литералы и комментарии, и печатает
места, где глубина уходит в минус или где закрывающих скобок подряд больше,
чем открывающих в той же строке.

Запуск:
    uv run python scripts/check_pine_parens.py pine/mr_quant_v2.pine
"""
from __future__ import annotations

import sys
from pathlib import Path


def strip_noise(line: str) -> str:
    """Убирает комментарий и содержимое строковых литералов."""
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch == "/" and i + 1 < n and line[i + 1] == "/":
            break
        if ch == '"':
            i += 1
            while i < n and line[i] != '"':
                i += 2 if line[i] == "\\" else 1
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def quote_balance(line: str) -> int:
    """Число кавычек в строке с учётом экранирования."""
    count, i, n = 0, 0, len(line)
    while i < n:
        if line[i] == '"' and (i == 0 or line[i - 1] != "\\"):
            count += 1
        i += 1
    return count


def main(path: str) -> int:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    depth = 0
    problems = 0

    print("проверка непарных кавычек (нечётное число = разъезд лексера):")
    for ln, raw in enumerate(lines, 1):
        code = strip_noise(raw)
        # strip_noise вырезает литералы; если кавычек в строке нечётное
        # число, литерал не закрылся и всё после него Pine считает строкой.
        q = quote_balance(raw)
        if q % 2 == 1:
            print(f"  СТРОКА {ln}: кавычек {q} (нечётно)")
            print(f"    {raw.rstrip()}")
            problems += 1
    if not problems:
        print("  непарных кавычек не найдено")

    for ln, raw in enumerate(lines, 1):
        code = strip_noise(raw)
        before = depth
        for ch in code:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                print(f"  СТРОКА {ln}: глубина ушла в минус (лишняя ')')")
                print(f"    {raw.rstrip()}")
                problems += 1
                depth = 0
                break

        # Строка закрывает больше скобок, чем открывает, и при этом НЕ
        # продолжает выражение — подозрительно.
        opens = code.count("(")
        closes = code.count(")")
        if opens == 0 and closes > 0 and before == 0:
            print(f"  СТРОКА {ln}: закрывающих скобок {closes}, открывающих 0, "
                  f"глубина до строки была 0")
            print(f"    {raw.rstrip()}")
            problems += 1

        if ")))" in code.replace(" ", ""):
            print(f"  СТРОКА {ln}: три закрывающих подряд — {raw.strip()[:80]}")

    print(f"\nитоговая глубина: {depth} (должно быть 0)")
    print(f"подозрительных строк: {problems}")
    return 1 if (depth != 0 or problems) else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pine/mr_quant_v2.pine"
    raise SystemExit(main(target))
