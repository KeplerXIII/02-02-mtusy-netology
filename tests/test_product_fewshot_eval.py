"""Продуктовый тест: для каждого кейса два вызова API (с few-shot и без), отчёт, рубрика на обоих."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.product]


@pytest.mark.skipif(
    not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="Нет DEEPSEEK_API_KEY / OPENAI_API_KEY",
)
def test_product_fewshot_eval_writes_report_and_passes_rubric(tmp_path: Path) -> None:
    from src.product_fewshot_eval import run_product_eval, write_report

    run = run_product_eval()
    report_path = tmp_path / "product_eval_fewshots.md"
    write_report(run, report_path)
    assert report_path.is_file()
    text = report_path.read_text(encoding="utf-8")
    assert "few-shot" in text.lower() and "без" in text.lower()
    assert "Вариант A" in text and "Вариант B" in text
    assert "| `fs_password` |" in text
    assert "IGNORE ALL PREVIOUS" in text or "injection" in text.lower()
    assert run.all_pass, (
        "Рубрика не прошла для: "
        + ", ".join(
            f"{c.case_id}({which})"
            for c in run.cases
            for which, ok in (
                ("FS", c.with_few_shot.rubric_pass),
                ("NO", c.without_few_shot.rubric_pass),
            )
            if not ok
        )
    )
