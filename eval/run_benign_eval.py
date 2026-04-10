from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.detect_secret_patterns import PatternDetector
from eval.evaluators import HFRunner, OllamaRunner, summarize_benign_results, write_report
from train.utils import load_yaml


def merge_overrides(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def load_benign_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


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
    parser = argparse.ArgumentParser(description="Run benign evaluation prompts against the selected backend.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_config.yaml"))
    parser.add_argument("--profile", type=str, default="lora")
    args = parser.parse_args()

    eval_config = load_yaml(args.config)
    if args.profile not in eval_config["profiles"]:
        raise KeyError(f"Unknown profile: {args.profile}")
    profile_config = eval_config["profiles"][args.profile]
    project_config = load_yaml(Path(profile_config.get("project_config", eval_config["project_config"])))
    root = Path(project_config["project"]["root_dir"]).resolve()
    detector = PatternDetector.from_files(root=root, safety_config_path=Path(project_config["safety"]["patterns_config"]))
    runtime_config = merge_overrides(eval_config["runtime"], profile_config.get("runtime_override"))
    runner = build_runner(profile_config, project_config, root, timeout_seconds=int(runtime_config["timeout_seconds"]))

    benign_records = load_benign_records(root / project_config["dataset"]["benign_eval_path"])
    system_prompt = ""
    if profile_config.get("use_system_template"):
        system_prompt = (root / project_config["templates"]["system_prompt"]).read_text(encoding="utf-8").strip()

    results: list[dict[str, Any]] = []
    for item in benign_records:
        user_prompt = item["messages"][0]["content"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
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
        leak = detector.scan(response)
        results.append({**item, "response": response, "leak": leak})

    summary = summarize_benign_results(results, detector.refusal_markers)
    payload = {"profile": args.profile, "summary": summary, "results": results}
    output_path = root / profile_config.get("benign_output_path", eval_config["benign_eval"]["output_path"])
    write_report(output_path, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
