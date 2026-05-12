#!/usr/bin/env python3
"""
Продуктовый прогон: 5 вопросов по темам few-shot — для каждого **два** запроса
(system **с** примерами и **без**), токены API + tiktoken, рубрика на обоих ответах, отчёт Markdown.

  uv run python scripts/product_eval_fewshots.py
  uv run python scripts/product_eval_fewshots.py --out reports/my_run.md

Итого **10** вызовов API. Нужен .env с DEEPSEEK_API_KEY (или переменные окружения).
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=False)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlib import Path

from src.product_fewshot_eval import (
    default_report_path,
    run_product_eval,
    write_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Продуктовый отчёт few-shot")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Путь к .md (по умолчанию reports/product_eval_fewshots_<UTC>.md)",
    )
    args = ap.parse_args()

    try:
        run = run_product_eval()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    path = args.out or default_report_path(Path(ROOT))
    write_report(run, path)
    print(path.resolve())
    print("Итог рубрики:", "PASS (все кейсы)" if run.all_pass else "FAIL (см. отчёт)")
    return 0 if run.all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
