#!/usr/bin/env python3
"""
Интерактивный чат с ассистентом техподдержки (system prompt из шаблона + DeepSeek).

  uv run python scripts/chat_support.py

Ключ и модель — из .env или окружения (см. README). Выход: пустая строка, /q, /quit, Ctrl+D.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"), override=False)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openai import OpenAI

from src.llm_config import deepseek_client
from src.prompt_service import render_tech_support_system_prompt


def _print_help() -> None:
    print(
        "Команды: /q или /quit — выход; /clear — сброс истории диалога.\n"
        "Пишите сообщения пользователю поддержки (Enter — отправить).\n",
        flush=True,
    )


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Чат техподдержки (DeepSeek)")
    p.add_argument("--no-few-shot", action="store_true", help="System prompt без примеров")
    p.add_argument("--cot", action="store_true", help="System prompt с инструкцией CoT")
    args = p.parse_args()

    try:
        client, model = deepseek_client()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        print("Создайте .env из .env.example или задайте DEEPSEEK_API_KEY.", file=sys.stderr)
        return 1

    system = render_tech_support_system_prompt(
        include_few_shot=not args.no_few_shot,
        include_chain_of_thought=args.cot,
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    print(f"Модель: {model}. System: few-shot={not args.no_few_shot}, cot={args.cot}\n")
    _print_help()

    while True:
        try:
            line = input("Вы: ").strip()
        except EOFError:
            print()
            break
        if not line or line in ("/q", "/quit", "/exit"):
            break
        if line == "/clear":
            messages = [{"role": "system", "content": system}]
            print("История очищена.\n", flush=True)
            continue
        if line == "/help":
            _print_help()
            continue

        messages.append({"role": "user", "content": line})
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=8192,
            )
        except Exception as ex:
            print(f"Ошибка API: {ex}\n", flush=True)
            messages.pop()
            continue

        reply = (r.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": reply})
        print(f"\nПоддержка:\n{reply}\n", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
