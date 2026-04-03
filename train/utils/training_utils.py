from __future__ import annotations

import ast
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return ast.literal_eval(value)
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if lines[index][1].startswith("- "):
        items: list[Any] = []
        while index < len(lines):
            current_indent, stripped = lines[index]
            if current_indent != indent or not stripped.startswith("- "):
                break
            body = stripped[2:].strip()
            if not body:
                value, index = _parse_block(lines, index + 1, indent + 2)
                items.append(value)
                continue
            if ":" in body:
                key, value_part = body.split(":", 1)
                item: dict[str, Any] = {key.strip(): _parse_scalar(value_part.strip()) if value_part.strip() else {}}
                index += 1
                while index < len(lines):
                    next_indent, next_stripped = lines[index]
                    if next_indent < indent + 2 or (next_indent == indent and next_stripped.startswith("- ")):
                        break
                    if next_indent == indent + 2:
                        if next_stripped.endswith(":"):
                            nested_key = next_stripped[:-1].strip()
                            nested_value, index = _parse_block(lines, index + 1, indent + 4)
                            item[nested_key] = nested_value
                        else:
                            nested_key, nested_part = next_stripped.split(":", 1)
                            item[nested_key.strip()] = _parse_scalar(nested_part.strip())
                            index += 1
                    else:
                        break
                items.append(item)
            else:
                items.append(_parse_scalar(body))
                index += 1
        return items, index

    mapping: dict[str, Any] = {}
    while index < len(lines):
        current_indent, stripped = lines[index]
        if current_indent != indent or stripped.startswith("- "):
            break
        if stripped.endswith(":"):
            key = stripped[:-1].strip()
            value, index = _parse_block(lines, index + 1, indent + 2)
            mapping[key] = value
            continue
        key, value_part = stripped.split(":", 1)
        mapping[key.strip()] = _parse_scalar(value_part.strip())
        index += 1
    return mapping, index


def _simple_yaml_load(text: str) -> dict[str, Any]:
    lines = [
        (len(raw_line) - len(raw_line.lstrip(" ")), raw_line.strip())
        for raw_line in text.splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
    ]
    if not lines:
        return {}
    parsed, _ = _parse_block(lines, 0, lines[0][0])
    return parsed


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml_load(text)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(log_path.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
