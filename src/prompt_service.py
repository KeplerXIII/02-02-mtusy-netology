from __future__ import annotations

from pathlib import Path
from typing import Any

import tiktoken
from jinja2 import Environment, FileSystemLoader, select_autoescape

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
TEMPLATE_NAME = "tech_support_system.jinja2"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_tech_support_system_prompt(
    *,
    product_name: str = "CloudDocs",
    support_hours: str = "пн–пт, 09:00–18:00 (МСК)",
    include_few_shot: bool = True,
    include_chain_of_thought: bool = False,
    **extra: Any,
) -> str:
    """Рендерит system prompt из Jinja2-шаблона."""
    tpl = _env().get_template(TEMPLATE_NAME)
    return tpl.render(
        product_name=product_name,
        support_hours=support_hours,
        include_few_shot=include_few_shot,
        include_chain_of_thought=include_chain_of_thought,
        **extra,
    )


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Подсчёт токенов (кодировка как у GPT-4 / cl100k_base)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def assert_token_range(
    text: str,
    low: int = 300,
    high: int = 800,
    model: str = "gpt-4",
) -> int:
    """Возвращает число токенов; используется в тестах и скриптах."""
    n = count_tokens(text, model=model)
    if not low <= n <= high:
        raise ValueError(f"Ожидалось {low}–{high} токенов, получено {n}")
    return n
