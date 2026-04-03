from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def fallback_chat_format(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    rendered: list[str] = []
    for message in messages:
        role = message["role"].upper()
        content = message["content"].strip()
        rendered.append(f"[{role}]\n{content}")
    if add_generation_prompt:
        rendered.append("[ASSISTANT]")
    return "\n\n".join(rendered)


def apply_chat_template_to_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return fallback_chat_format(messages, add_generation_prompt=add_generation_prompt)
    return fallback_chat_format(messages, add_generation_prompt=add_generation_prompt)


def build_text_dataset(path: Path, tokenizer: Any) -> Dataset:
    rows = read_jsonl(path)
    formatted_rows = [
        {"text": apply_chat_template_to_messages(tokenizer, row["messages"], add_generation_prompt=False)}
        for row in rows
    ]
    return Dataset.from_list(formatted_rows)
