from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.utils import load_yaml


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "summary": {}, "results": []}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline, secure Modelfile, and LoRA evaluation reports.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval_config.yaml"))
    args = parser.parse_args()

    eval_config = load_yaml(args.config)
    project_config = load_yaml(Path(eval_config["project_config"]))
    root = Path(project_config["project"]["root_dir"]).resolve()

    baseline = load_report(root / eval_config["profiles"]["baseline"]["output_path"])
    secure = load_report(root / eval_config["profiles"]["secure_modelfile"]["output_path"])
    lora = load_report(root / eval_config["profiles"]["lora"]["output_path"])
    benign = load_report(
        root / eval_config["profiles"]["lora"].get("benign_output_path", eval_config["benign_eval"]["output_path"])
    )

    comparison = {
        "attack_profiles": {
            "baseline": baseline.get("summary", {}),
            "secure_modelfile": secure.get("summary", {}),
            "lora": lora.get("summary", {}),
        },
        "benign_profile": benign.get("summary", {}),
        "delta_notes": [
            "Compare leakage_rate and fail counts between baseline and secure_modelfile.",
            "Check whether LoRA keeps benign success while reducing attack fail cases.",
            "Review category_summary deltas to identify residual weak spots.",
        ],
    }

    output_path = root / eval_config["comparison"]["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
