from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_exact_values(raw_path: Path) -> set[str]:
    values: set[str] = set()
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            continue
        if "=" in stripped:
            _, value = stripped.split("=", 1)
            values.add(value.strip())
        if ":" in stripped:
            _, value = stripped.split(":", 1)
            candidate = value.strip().lstrip("-").strip()
            if candidate and len(candidate) > 6:
                values.add(candidate)
    return {item for item in values if " " not in item or item.startswith("-----BEGIN")}


def build_patterns(root: Path, config: dict[str, Any]) -> tuple[list[re.Pattern[str]], set[str]]:
    safety = load_yaml(root / config["safety"]["patterns_config"])
    regexes = [re.compile(item["regex"]) for item in safety["leakage_patterns"]]
    exact_values: set[str] = set()
    for rel_path in safety.get("exact_match_sources", []):
        source = root / rel_path
        if source.exists():
            exact_values.update(extract_exact_values(source))
    return regexes, exact_values


def validate_messages(record: dict[str, Any], path: Path, row_index: int) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append(f"{path}:{row_index} messages must contain at least system/user/assistant triplet")
        return errors
    expected_roles = ["system", "user", "assistant"]
    for idx, expected_role in enumerate(expected_roles):
        message = messages[idx]
        if message.get("role") != expected_role:
            errors.append(f"{path}:{row_index} expected role {expected_role} at position {idx}")
        if not str(message.get("content", "")).strip():
            errors.append(f"{path}:{row_index} empty content at position {idx}")
    return errors


def validate_assistant_content(
    record: dict[str, Any],
    path: Path,
    row_index: int,
    regexes: list[re.Pattern[str]],
    exact_values: set[str],
) -> list[str]:
    errors: list[str] = []
    assistant_text = str(record["messages"][2].get("content", ""))
    for pattern in regexes:
        if pattern.search(assistant_text):
            errors.append(f"{path}:{row_index} assistant content matched leakage regex: {pattern.pattern}")
    for value in exact_values:
        if value and value in assistant_text:
            errors.append(f"{path}:{row_index} assistant content contains exact dummy value: {value}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate JSONL format and detect accidental secret-like outputs in the processed dataset."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.yaml"))
    parser.add_argument("--paths", nargs="*", default=None, help="Optional JSONL paths to validate. Defaults to train/valid.")
    args = parser.parse_args()

    project_config = load_yaml(args.config)
    root = Path(project_config["project"]["root_dir"]).resolve()
    default_paths = [root / project_config["dataset"]["train_path"], root / project_config["dataset"]["valid_path"]]
    target_paths = [Path(path).resolve() for path in args.paths] if args.paths else default_paths
    regexes, exact_values = build_patterns(root, project_config)

    errors: list[str] = []
    inspected_rows = 0
    for path in target_paths:
        if not path.exists():
            errors.append(f"missing dataset file: {path}")
            continue
        for row_index, record in enumerate(read_jsonl(path), start=1):
            inspected_rows += 1
            errors.extend(validate_messages(record, path, row_index))
            if "messages" in record and len(record["messages"]) >= 3:
                errors.extend(validate_assistant_content(record, path, row_index, regexes, exact_values))

    if errors:
        print(json.dumps({"status": "failed", "errors": errors, "rows_checked": inspected_rows}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print(
        json.dumps(
            {"status": "ok", "rows_checked": inspected_rows, "checked_files": [str(path) for path in target_paths]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
