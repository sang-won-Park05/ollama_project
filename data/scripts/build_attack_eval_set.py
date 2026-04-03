from __future__ import annotations

import argparse
import ast
import json
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
            items.append(_parse_scalar(stripped[2:].strip()))
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build attack evaluation JSONL from prompt files and manifest metadata.")
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.yaml"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    root = Path(config["project"]["root_dir"]).resolve()
    attacks_root = root / config["paths"]["attacks_root"]
    output_path = root / config["dataset"]["attack_eval_path"]
    manifest = json.loads((attacks_root / "attack_manifest.json").read_text(encoding="utf-8"))

    context_files = [
        "data/raw/dummy.env",
        "data/raw/dummy_secrets.txt",
        "data/raw/dummy_internal_prompt.txt",
        "data/raw/dummy_policy.txt",
        "data/raw/dummy_incident_note.txt",
        "data/raw/dummy_mixed_document.txt",
    ]
    rows: list[dict[str, Any]] = []
    for item in manifest:
        prompt_path = attacks_root / item["filename"]
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "filename": item["filename"],
                "objective": item["objective"],
                "severity": item["severity"],
                "tags": item["tags"],
                "context_files": context_files,
                "messages": [{"role": "user", "content": prompt_path.read_text(encoding="utf-8").strip()}],
            }
        )

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} attack eval records to {output_path}")


if __name__ == "__main__":
    main()
