from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.utils import load_yaml, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export metadata that will later be used for GGUF or Ollama packaging.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    project_config = load_yaml(Path(train_config["project_config"]))
    root = Path(project_config["project"]["root_dir"]).resolve()

    stats_path = root / project_config["dataset"]["stats_path"]
    dataset_stats = {}
    if stats_path.exists():
        dataset_stats = load_yaml(stats_path) if stats_path.suffix in {".yaml", ".yml"} else {}
        if not dataset_stats:
            import json

            dataset_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    payload = {
        "project": project_config["project"]["name"],
        "base_model_id": project_config["model"]["base_model_id"],
        "adapter_dir": train_config["training"]["output_adapter_dir"],
        "merged_dir": "outputs/merged/exaone-security-merged",
        "gguf_target": "outputs/gguf/exaone-security.gguf",
        "dataset_stats": dataset_stats,
        "lora": train_config["lora"],
        "generation_defaults": train_config["generation"],
        "todo": [
            "TODO: Convert the merged model to GGUF using your selected converter.",
            "TODO: Replace placeholder Ollama FROM/ADAPTER paths in modelfiles before packaging.",
        ],
    }
    save_json(root / "outputs" / "gguf" / "export_manifest.json", payload)
    print("Saved export metadata to outputs/gguf/export_manifest.json")


if __name__ == "__main__":
    main()
