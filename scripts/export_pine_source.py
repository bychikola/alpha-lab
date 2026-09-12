"""Кладёт исходник индикатора в полигон как JS-переменную.

Полигон открывается через file://, а fetch() к локальным файлам браузер
блокирует. Поэтому исходник подключается скриптом, как и данные превью.

Запуск:
    uv run python scripts/export_pine_source.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pine" / "mr_quant_v2.pine"
OUT = ROOT / "polygon" / "pine-source.js"


def main() -> int:
    if not SRC.exists():
        print(f"нет файла {SRC}", file=sys.stderr)
        return 1
    text = SRC.read_text(encoding="utf-8")

    if "`" in text:
        # Обратная кавычка сломала бы шаблонный литерал в JS.
        print("в исходнике есть обратная кавычка — экранируйте вручную",
              file=sys.stderr)
        return 1

    OUT.write_text(
        "// Сгенерировано scripts/export_pine_source.py из pine/mr_quant_v2.pine\n"
        "window.PINE_SOURCE = " + json.dumps(text, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"записано: {OUT} ({OUT.stat().st_size / 1024:.0f} КБ, "
          f"{len(text.splitlines())} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
