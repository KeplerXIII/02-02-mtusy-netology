"""Один «живой» вызов DeepSeek при наличии DEEPSEEK_API_KEY; без ключа тест пропускается."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="Нет DEEPSEEK_API_KEY / OPENAI_API_KEY",
)
def test_deepseek_chat_minimal() -> None:
    from openai import OpenAI

    from src.llm_config import DEFAULT_DEEPSEEK_BASE, DEFAULT_DEEPSEEK_MODEL

    key = (
        os.environ["DEEPSEEK_API_KEY"]
        if os.environ.get("DEEPSEEK_API_KEY")
        else os.environ["OPENAI_API_KEY"]
    )
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    base = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE).rstrip("/")
    client = OpenAI(api_key=key, base_url=base)
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Отвечай одним словом: да или нет."},
            {"role": "user", "content": "2+2=4?"},
        ],
        temperature=0,
        max_tokens=512,
    )
    text = (r.choices[0].message.content or "").strip().lower()
    assert "да" in text or "yes" in text or "4" in text or len(text) > 0
