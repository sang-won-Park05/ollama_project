from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.utils import load_tokenizer, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the trained LoRA adapter into the base model.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    project_config = load_yaml(Path(train_config["project_config"]))
    root = Path(project_config["project"]["root_dir"]).resolve()

    base_model_id = project_config["model"]["base_model_id"]
    adapter_dir = root / train_config["training"]["output_adapter_dir"]
    merged_dir = root / "outputs" / "merged" / "exaone-security-merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(base_model_id, trust_remote_code=project_config["model"]["trust_remote_code"])
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=project_config["model"]["trust_remote_code"],
        torch_dtype=torch.bfloat16 if project_config["model"]["bf16"] else torch.float16,
        device_map="auto",
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
