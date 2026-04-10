from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.detect_secret_patterns import PatternDetector
from eval.evaluators import HFRunner, OllamaRunner, summarize_attack_results, write_report
from train.utils import has_refusal_marker, load_yaml, simple_attack_verdict


def merge_overrides(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def build_context(root: Path, context_config: dict[str, Any]) -> str:
    if not context_config["enabled"]:
        return ""
    delimiter = context_config["delimiter"]
    chunks: list[str] = []
    for rel_path in context_config["include_files"]:
        path = root / rel_path
        if not path.exists():
            continue
        chunks.append(f"[FILE] {rel_path}\n{path.read_text(encoding='utf-8').strip()}")
    return delimiter.join(chunks)


def load_attack_records(root: Path) -> list[dict[str, Any]]:
    attacks_root = root / "attacks"
    manifest = json.loads((attacks_root / "attack_manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for item in manifest:
        prompt_path = attacks_root / item["filename"]
        records.append({**item, "prompt": prompt_path.read_text(encoding="utf-8").strip()})
    return records


def select_attack_records(records: list[dict[str, Any]], profile_config: dict[str, Any]) -> list[dict[str, Any]]:
    selection = profile_config.get("attack_selection", {})
    attack_ids = selection.get("attack_ids")
    if attack_ids:
        record_map = {record["id"]: record for record in records}
        selected = [record_map[attack_id] for attack_id in attack_ids if attack_id in record_map]
    else:
        selected = list(records)
    max_attacks = selection.get("max_attacks")
    if max_attacks is not None:
        selected = selected[: int(max_attacks)]
    return selected


def load_existing_results(output_path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    results = payload.get("results", [])
    return [item for item in results if item.get("id") in selected_ids]


def build_payload(
    *,
    profile_name: str,
    backend: str,
    runtime_config: dict[str, Any],
    selected_records: list[dict[str, Any]],
    results: list[dict[str, Any]],
    refusal_markers: list[str],
) -> dict[str, Any]:
    summary = summarize_attack_results(results, refusal_markers)
    return {
        "profile": profile_name,
        "backend": backend,
        "runtime": runtime_config,
        "progress": {
            "completed": len(results),
            "total": len(selected_records),
            "remaining": max(len(selected_records) - len(results), 0),
        },
        "summary": summary,
        "results": results,
    }


def build_runner(
    profile_config: dict[str, Any],
    project_config: dict[str, Any],
    root: Path,
    timeout_seconds: int,
) -> HFRunner | OllamaRunner:
    backend = profile_config["backend"]
    if backend == "hf":
        adapter_path = profile_config["hf"]["adapter_path"]
        resolved_adapter = str((root / adapter_path).resolve()) if adapter_path else None
        return HFRunner(
            model_name_or_path=profile_config["hf"]["model_name_or_path"],
            adapter_path=resolved_adapter,
            trust_remote_code=project_config["model"]["trust_remote_code"],
            load_in_4bit=project_config["model"]["load_in_4bit"],
            use_bf16=project_config["model"]["bf16"],
            device_preference=project_config["model"].get("device"),
            dtype_name=project_config["model"].get("dtype"),
            use_mps=bool(project_config["model"].get("use_mps", False)),
        )
    return OllamaRunner(
        model=profile_config["ollama"]["model"],
        mode=profile_config["ollama"]["mode"],
        endpoint=profile_config["ollama"]["endpoint"],
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attack evaluation using either HF or Ollama backend.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_config.yaml"))
    parser.add_argument("--profile", type=str, default="baseline")
    args = parser.parse_args()

    eval_config = load_yaml(args.config)
    if args.profile not in eval_config["profiles"]:
        raise KeyError(f"Unknown profile: {args.profile}")
    profile_config = eval_config["profiles"][args.profile]
    project_config = load_yaml(Path(profile_config.get("project_config", eval_config["project_config"])))
    root = Path(project_config["project"]["root_dir"]).resolve()
    detector = PatternDetector.from_files(root=root, safety_config_path=Path(project_config["safety"]["patterns_config"]))
    refusal_markers = detector.refusal_markers

    runtime_config = merge_overrides(eval_config["runtime"], profile_config.get("runtime_override"))
    context_config = merge_overrides(eval_config["context"], profile_config.get("context_override"))
    runner = build_runner(profile_config, project_config, root, timeout_seconds=int(runtime_config["timeout_seconds"]))

    system_prompt = ""
    if profile_config.get("use_system_template"):
        system_prompt = (root / project_config["templates"]["system_prompt"]).read_text(encoding="utf-8").strip()
    context_block = build_context(root, context_config)

    selected_records = select_attack_records(load_attack_records(root), profile_config)
    total_attacks = len(selected_records)
    output_path = root / profile_config["output_path"]
    resume_enabled = bool(profile_config.get("resume", runtime_config.get("resume", True)))
    save_every = int(profile_config.get("save_every", runtime_config.get("save_every", 1)))
    order_map = {record["id"]: index for index, record in enumerate(selected_records)}
    selected_ids = set(order_map)
    results = load_existing_results(output_path, selected_ids) if resume_enabled else []
    results.sort(key=lambda item: order_map.get(item["id"], total_attacks))
    completed_ids = {item["id"] for item in results}

    if completed_ids:
        print(
            f"[RESUME] profile={args.profile} completed={len(completed_ids)}/{total_attacks} output={output_path}",
            flush=True,
        )

    processed_since_save = 0
    for index, record in enumerate(selected_records, start=1):
        if record["id"] in completed_ids:
            print(f"[SKIP {index}/{total_attacks}] {record['id']} already saved", flush=True)
            continue

        attack_start = time.perf_counter()
        print(
            f"[START {index}/{total_attacks}] id={record['id']} category={record['category']}",
            flush=True,
        )
        user_prompt = record["prompt"]
        if context_block:
            user_prompt = f"{user_prompt}\n\n[LOCAL_TEST_CONTEXT]\n{context_block}"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = ""
        error: str | None = None
        try:
            response = runner.generate(
                messages,
                max_new_tokens=int(runtime_config["max_new_tokens"]),
                temperature=float(runtime_config["temperature"]),
                top_p=float(runtime_config["top_p"]),
                do_sample=False,
                timeout_seconds=int(runtime_config["timeout_seconds"]),
                max_input_tokens=(
                    int(runtime_config["max_input_tokens"])
                    if runtime_config.get("max_input_tokens") not in (None, "")
                    else None
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive path for long local eval runs
            error = f"{type(exc).__name__}: {exc}"

        elapsed_seconds = round(time.perf_counter() - attack_start, 3)
        leak = detector.scan(response)
        refusal_found = has_refusal_marker(response, refusal_markers)
        verdict = simple_attack_verdict(len(leak["hits"]), refusal_found)
        result: dict[str, Any] = {
            **record,
            "response": response,
            "leak": leak,
            "verdict": verdict,
            "elapsed_seconds": elapsed_seconds,
        }
        if error:
            result["error"] = error
        results.append(result)
        results.sort(key=lambda item: order_map.get(item["id"], total_attacks))
        completed_ids.add(record["id"])
        processed_since_save += 1

        print(
            f"[DONE {index}/{total_attacks}] id={record['id']} elapsed={elapsed_seconds:.3f}s verdict={verdict} leaks={len(leak['hits'])}",
            flush=True,
        )
        if error:
            print(f"[ERROR {index}/{total_attacks}] id={record['id']} detail={error}", flush=True)

        if processed_since_save >= save_every:
            payload = build_payload(
                profile_name=args.profile,
                backend=profile_config["backend"],
                runtime_config=runtime_config,
                selected_records=selected_records,
                results=results,
                refusal_markers=refusal_markers,
            )
            write_report(output_path, payload)
            processed_since_save = 0

    payload = build_payload(
        profile_name=args.profile,
        backend=profile_config["backend"],
        runtime_config=runtime_config,
        selected_records=selected_records,
        results=results,
        refusal_markers=refusal_markers,
    )
    write_report(output_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
