from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.utils import infer_target_modules, load_causal_lm, load_tokenizer, load_yaml, save_json, setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model modules and recommend LoRA target modules.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    project_config = load_yaml(Path(config["project_config"]))
    root = Path(project_config["project"]["root_dir"]).resolve()
    logging_dir = root / config["training"].get("logging_dir", "logs/train")
    debug_dir = root / config["training"].get("debug_dir", "logs/debug")
    logger = setup_logger(logging_dir / "inspect_modules.log")

    model_name = project_config["model"]["base_model_id"]
    logger.info("Loading tokenizer and model for module inspection: %s", model_name)
    _ = load_tokenizer(model_name, trust_remote_code=project_config["model"]["trust_remote_code"])
    model = load_causal_lm(
        model_name,
        trust_remote_code=project_config["model"]["trust_remote_code"],
        load_in_4bit=project_config["model"]["load_in_4bit"],
        use_bf16=project_config["model"]["bf16"],
        gradient_checkpointing=False,
        device_preference=project_config["model"].get("device"),
        dtype_name=project_config["model"].get("dtype"),
        use_mps=bool(project_config["model"].get("use_mps", False)),
    )

    module_lines: list[str] = []
    suffix_counter: Counter[str] = Counter()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        module_lines.append(f"{name}\t{class_name}")
        suffix_counter[name.rsplit(".", 1)[-1]] += 1

    output_path = debug_dir / "module_names.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(module_lines) + "\n", encoding="utf-8")

    preferred_targets = list(config["lora"]["target_modules"])
    recommended = infer_target_modules(model, preferred_targets)
    summary = {
        "preferred_targets": preferred_targets,
        "recommended_targets": recommended,
        "top_suffixes": suffix_counter.most_common(20),
    }
    save_json(debug_dir / "module_summary.json", summary)

    logger.info("Saved module list to %s", output_path)
    logger.info("Recommended target modules: %s", ", ".join(recommended))
    print("Recommended target modules:", ", ".join(recommended))


if __name__ == "__main__":
    main()
