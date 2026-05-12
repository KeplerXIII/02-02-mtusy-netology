"""Настройки LLM: DeepSeek (OpenAI-совместимый API). Ключи только из окружения."""

from __future__ import annotations

import os

from openai import OpenAI

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"


def deepseek_client() -> tuple[OpenAI, str]:
    """
    Клиент и имя модели.
    Ключ: DEEPSEEK_API_KEY (или OPENAI_API_KEY для обратной совместимости).
    """
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Задайте DEEPSEEK_API_KEY (или OPENAI_API_KEY) в окружении.",
        )
    base = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE).rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    client = OpenAI(api_key=key, base_url=base)
    return client, model
