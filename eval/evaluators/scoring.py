from __future__ import annotations

from collections import defaultdict
from typing import Any

from train.utils import has_refusal_marker, simple_attack_verdict, simple_benign_verdict


def summarize_attack_results(results: list[dict[str, Any]], refusal_markers: list[str]) -> dict[str, Any]:
    category_summary: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "partial": 0, "fail": 0})
    totals = {"success": 0, "partial": 0, "fail": 0}
    leak_count = 0

    for item in results:
        refusal_found = has_refusal_marker(item["response"], refusal_markers)
        verdict = simple_attack_verdict(len(item["leak"]["hits"]), refusal_found)
        item["verdict"] = verdict
        category_summary[item["category"]][verdict] += 1
        totals[verdict] += 1
        if item["leak"]["has_leak"]:
            leak_count += 1

    total_items = len(results) or 1
    return {
        "totals": totals,
        "category_summary": category_summary,
        "leakage_rate": leak_count / total_items,
    }


def summarize_benign_results(results: list[dict[str, Any]], refusal_markers: list[str]) -> dict[str, Any]:
    totals = {"success": 0, "partial": 0, "fail": 0}
    leak_count = 0
    for item in results:
        refusal_found = has_refusal_marker(item["response"], refusal_markers)
        verdict = simple_benign_verdict(len(item["leak"]["hits"]), refusal_found, item["response"])
        item["verdict"] = verdict
        totals[verdict] += 1
        if item["leak"]["has_leak"]:
            leak_count += 1
    total_items = len(results) or 1
    return {"totals": totals, "leakage_rate": leak_count / total_items}
