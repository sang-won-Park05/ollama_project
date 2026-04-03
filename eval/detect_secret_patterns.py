from __future__ import annotations

import ast
import re
from dataclasses import dataclass
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


@dataclass
class CompiledPattern:
    name: str
    severity: str
    pattern: re.Pattern[str]


class PatternDetector:
    def __init__(
        self,
        compiled_patterns: list[CompiledPattern],
        exact_values: set[str],
        refusal_markers: list[str],
    ) -> None:
        self.compiled_patterns = compiled_patterns
        self.exact_values = {value for value in exact_values if value}
        self.refusal_markers = refusal_markers

    @classmethod
    def from_files(cls, root: Path, safety_config_path: Path) -> "PatternDetector":
        config_path = safety_config_path if safety_config_path.is_absolute() else root / safety_config_path
        text = config_path.read_text(encoding="utf-8")
        safety = yaml.safe_load(text) if yaml is not None else _simple_yaml_load(text)
        compiled_patterns = [
            CompiledPattern(
                name=item["name"],
                severity=item["severity"],
                pattern=re.compile(item["regex"]),
            )
            for item in safety["leakage_patterns"]
        ]
        exact_values: set[str] = set()
        for rel_path in safety.get("exact_match_sources", []):
            exact_values.update(_extract_values_from_file(root / rel_path))
        refusal_markers = list(safety["refusal_markers"]["ko"]) + list(safety["refusal_markers"]["en"])
        return cls(compiled_patterns=compiled_patterns, exact_values=exact_values, refusal_markers=refusal_markers)

    def scan(self, text: str) -> dict[str, Any]:
        hits: list[dict[str, str]] = []
        for item in self.compiled_patterns:
            for match in item.pattern.finditer(text):
                hits.append(
                    {
                        "type": "regex",
                        "name": item.name,
                        "severity": item.severity,
                        "match": match.group(0),
                    }
                )
        for value in sorted(self.exact_values):
            if value in text:
                hits.append(
                    {
                        "type": "exact",
                        "name": "exact_dummy_value",
                        "severity": "high",
                        "match": value,
                    }
                )
        return {"has_leak": bool(hits), "hit_count": len(hits), "hits": hits}


def _extract_values_from_file(path: Path) -> set[str]:
    values: set[str] = set()
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            continue
        if "=" in stripped:
            _, value = stripped.split("=", 1)
            values.add(value.strip())
        if ":" in stripped:
            _, value = stripped.split(":", 1)
            candidate = value.strip().lstrip("-").strip()
            if candidate and len(candidate) > 6 and " " not in candidate:
                values.add(candidate)
    return values


def detect_secret_patterns_in_text(root: Path, safety_config_path: Path, text: str) -> dict[str, Any]:
    detector = PatternDetector.from_files(root=root, safety_config_path=safety_config_path)
    return detector.scan(text)
