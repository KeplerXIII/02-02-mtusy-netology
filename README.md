# 02-02-mtusy-netology

Зависимости и окружение — через [uv](https://docs.astral.sh/uv/).  
  
Готовый `.md` отчет лежит в папке [reports](reports/).  
JINJA-промт лежит в папке [prompts](prompts/).  
Список продуктовых кейсов (вопросы и рубрики): константа [`FEW_SHOT_CASES`](src/product_fewshot_eval.py#L134) в `src/product_fewshot_eval.py` (строки 134–174).  

## Установка и тесты

Локальные тесты **без сети** (промпт, tiktoken):

```bash
uv sync
uv run pytest
```

## Скрипт сравнения LLM (DeepSeek)

Используется OpenAI-совместимый клиент: `https://api.deepseek.com`, модель по умолчанию **`deepseek-v4-pro`**.

Переменные можно положить в **`.env`** в корне (файл в `.gitignore`, в репозиторий не попадает) — подхватываются `python-dotenv` в скрипте и в pytest.

```bash
uv sync
cp .env.example .env   # отредактируйте ключ
uv run python scripts/compare_fewshot_and_cot.py
```

Либо `export DEEPSEEK_API_KEY=…`. Поддерживается и `OPENAI_API_KEY`, если `DEEPSEEK_API_KEY` не задан.

Шаблон: `.env.example`.

## Чат в терминале

После `uv sync` и настроенного `.env`:

```bash
uv run python scripts/chat_support.py
```

Опции: `--no-few-shot`, `--cot`. Команды в чате: `/clear` — новый диалог; `/q` — выход.

## Продуктовый отчёт (few-shot **vs** без)

Пять вопросов по темам из примеров; на **каждый** вопрос два запроса к API: system **с** блоком примеров и **без** него (удобно сравнить ответы). В логе: оба ответа, **токены** из `usage` и **tiktoken**, рубрика PASS/FAIL **для каждого варианта**. Всего **10** вызовов API за прогон.

```bash
uv run python scripts/product_eval_fewshots.py
# отчёт: reports/product_eval_fewshots_<UTC>.md
# свой путь: uv run python scripts/product_eval_fewshots.py --out reports/отчёт.md
```

Тот же прогон в pytest (нужен ключ в `.env`):

```bash
uv run pytest tests/test_product_fewshot_eval.py -v -m product
```
