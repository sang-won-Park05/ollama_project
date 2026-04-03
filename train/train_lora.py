from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any
import sys

from transformers import TrainingArguments
from trl import SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.utils import (
    attach_lora_adapter,
    build_text_dataset,
    infer_target_modules,
    load_causal_lm,
    load_tokenizer,
    load_yaml,
    save_json,
    set_global_seed,
    setup_logger,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a security-focused LoRA adapter on the local refusal dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    project_config = load_yaml(Path(train_config["project_config"]))
    root = Path(project_config["project"]["root_dir"]).resolve()

    log_path = root / train_config["training"]["logging_dir"] / "train.log"
    logger = setup_logger(log_path)

    try:
        set_global_seed(int(train_config["training"]["seed"]))
        model_name = project_config["model"]["base_model_id"]
        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = load_tokenizer(model_name, trust_remote_code=project_config["model"]["trust_remote_code"])

        logger.info("Loading 4bit base model for QLoRA training")
        model = load_causal_lm(
            model_name,
            trust_remote_code=project_config["model"]["trust_remote_code"],
            load_in_4bit=project_config["model"]["load_in_4bit"],
            use_bf16=train_config["training"]["bf16"],
            gradient_checkpointing=train_config["training"]["gradient_checkpointing"],
        )

        preferred_targets = list(train_config["lora"]["target_modules"])
        target_modules = infer_target_modules(model, preferred_targets)
        logger.info("Using target modules: %s", ", ".join(target_modules))
        model = attach_lora_adapter(model, train_config["lora"], target_modules)
        model.print_trainable_parameters()

        train_dataset = build_text_dataset(root / train_config["training"]["train_file"], tokenizer)
        eval_dataset = build_text_dataset(root / train_config["training"]["valid_file"], tokenizer)
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        checkpoint_dir = root / train_config["training"]["checkpoint_dir"]
        adapter_dir = root / train_config["training"]["output_adapter_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=float(train_config["training"]["num_train_epochs"]),
            per_device_train_batch_size=int(train_config["training"]["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(train_config["training"]["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(train_config["training"]["gradient_accumulation_steps"]),
            learning_rate=float(train_config["training"]["learning_rate"]),
            lr_scheduler_type=str(train_config["training"]["lr_scheduler_type"]),
            warmup_ratio=float(train_config["training"]["warmup_ratio"]),
            weight_decay=float(train_config["training"]["weight_decay"]),
            logging_steps=int(train_config["training"]["logging_steps"]),
            save_steps=int(train_config["training"]["save_steps"]),
            eval_steps=int(train_config["training"]["eval_steps"]),
            save_total_limit=int(train_config["training"]["save_total_limit"]),
            bf16=bool(train_config["training"]["bf16"]),
            max_steps=int(train_config["training"]["max_steps"]),
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            report_to=[],
            remove_unused_columns=False,
            optim=str(train_config["training"]["optim"]),
            seed=int(train_config["training"]["seed"]),
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=int(train_config["training"]["max_seq_length"]),
            args=training_args,
            packing=False,
        )

        logger.info("Starting training")
        train_result = trainer.train()
        eval_metrics: dict[str, Any] = trainer.evaluate()

        logger.info("Saving adapter to %s", adapter_dir)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        save_json(root / "logs" / "train" / "loss_history.json", trainer.state.log_history)
        save_json(root / "logs" / "train" / "eval_during_train.json", eval_metrics)
        save_json(
            root / "logs" / "train" / "train_summary.json",
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "target_modules": target_modules,
                "adapter_dir": str(adapter_dir),
                "checkpoint_dir": str(checkpoint_dir),
            },
        )
        logger.info("Training complete")
    except Exception as exc:  # pragma: no cover - defensive logging for long-running jobs
        logger.error("Training failed: %s", exc)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
