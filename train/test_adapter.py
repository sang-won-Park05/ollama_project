from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.detect_secret_patterns import PatternDetector  # noqa: E402
from train.utils import (  # noqa: E402
    apply_chat_template_to_messages,
    has_refusal_marker,
    load_model_with_adapter,
    load_tokenizer,
    load_yaml,
    save_json,
    simple_attack_verdict,
    simple_benign_verdict,
)


def load_representative_attack_prompts(attacks_root: Path) -> list[dict[str, str]]:
    selected_files = [
        "direct/001_basic_secret_request.txt",
        "direct/007_first_last_chars_request.txt",
        "indirect_docs/003_find_api_key_in_doc.txt",
        "indirect_docs/009_convert_doc_to_table.txt",
        "unicode_obfuscation/004_base64_then_explain.txt",
        "unicode_obfuscation/008_reverse_string_reveal.txt",
        "role_confusion/001_you_are_security_auditor.txt",
        "role_confusion/007_switch_to_system_mode.txt",
        "manyshot/008_json_format_extraction.txt",
        "multimodal_prep/003_ocr_internal_doc.txt",
    ]
    return [
        {"id": path.replace("/", "_").replace(".txt", ""), "prompt": (attacks_root / path).read_text(encoding="utf-8").strip()}
        for path in selected_files
    ]


def load_benign_prompts(processed_root: Path) -> list[dict[str, Any]]:
    benign_path = processed_root / "benign_eval.jsonl"
    rows: list[dict[str, Any]] = []
    if benign_path.exists():
        for line in benign_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows[:3]


def generate_response(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = apply_chat_template_to_messages(tokenizer, messages, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            top_p=0.85,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = encoded["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a smoke test against the trained adapter using attack and benign prompts.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_config.yaml"))
    parser.add_argument("--profile", type=str, default="lora")
    args = parser.parse_args()

    eval_config = load_yaml(args.config)
    profile = eval_config["profiles"][args.profile]
    project_config = load_yaml(Path(profile.get("project_config", eval_config["project_config"])))
    detector = PatternDetector.from_files(
        root=Path(project_config["project"]["root_dir"]).resolve(),
        safety_config_path=Path(project_config["safety"]["patterns_config"]),
    )

    adapter_path = profile["hf"]["adapter_path"]
    if not adapter_path or not (ROOT / adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    tokenizer = load_tokenizer(
        profile["hf"]["model_name_or_path"],
        trust_remote_code=project_config["model"]["trust_remote_code"],
    )
    model = load_model_with_adapter(
        profile["hf"]["model_name_or_path"],
        str(ROOT / adapter_path),
        trust_remote_code=project_config["model"]["trust_remote_code"],
        load_in_4bit=project_config["model"]["load_in_4bit"],
        use_bf16=project_config["model"]["bf16"],
        device_preference=project_config["model"].get("device"),
        dtype_name=project_config["model"].get("dtype"),
        use_mps=bool(project_config["model"].get("use_mps", False)),
    )
    model.eval()

    system_prompt = (ROOT / project_config["templates"]["system_prompt"]).read_text(encoding="utf-8").strip()
    refusal_markers = detector.refusal_markers

    results: list[dict[str, Any]] = []

    attack_items = load_representative_attack_prompts(ROOT / "attacks")
    for idx, item in enumerate(attack_items, start=1):
        print(f"[ATTACK {idx}/{len(attack_items)}] {item['id']}", flush=True)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["prompt"]},
        ]
        response = generate_response(model, tokenizer, messages, max_new_tokens=int(eval_config["runtime"]["max_new_tokens"]))
        leak = detector.scan(response)
        refusal_found = has_refusal_marker(response, refusal_markers)
        results.append(
            {
                "id": item["id"],
                "kind": "attack",
                "prompt": item["prompt"],
                "response": response,
                "leak": leak,
                "verdict": simple_attack_verdict(len(leak["hits"]), refusal_found),
            }
        )

    benign_eval_path = ROOT / project_config["dataset"]["benign_eval_path"]
    benign_items = load_benign_prompts(benign_eval_path.parent)
    for idx, item in enumerate(benign_items, start=1):
        print(f"[BENIGN {idx}/{len(benign_items)}] {item['id']}", flush=True)

        prompt = item["messages"][0]["content"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = generate_response(model, tokenizer, messages, max_new_tokens=int(eval_config["runtime"]["max_new_tokens"]))
        leak = detector.scan(response)
        refusal_found = has_refusal_marker(response, refusal_markers)
        results.append(
            {
                "id": item["id"],
                "kind": "benign",
                "prompt": prompt,
                "response": response,
                "leak": leak,
                "verdict": simple_benign_verdict(len(leak["hits"]), refusal_found, response),
            }
        )

    attack_results = [item for item in results if item["kind"] == "attack"]
    benign_results = [item for item in results if item["kind"] == "benign"]
    summary = {
        "attack_success": sum(1 for item in attack_results if item["verdict"] == "success"),
        "attack_partial": sum(1 for item in attack_results if item["verdict"] == "partial"),
        "attack_fail": sum(1 for item in attack_results if item["verdict"] == "fail"),
        "benign_success": sum(1 for item in benign_results if item["verdict"] == "success"),
        "benign_partial": sum(1 for item in benign_results if item["verdict"] == "partial"),
        "benign_fail": sum(1 for item in benign_results if item["verdict"] == "fail"),
    }

    payload = {"profile": args.profile, "summary": summary, "results": results}
    output_path = ROOT / profile.get("smoke_test_output_path", profile["output_path"])
    save_json(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()