from __future__ import annotations

import argparse
import gc
import traceback
from inspect import signature
from pathlib import Path
from typing import Any
import sys

import torch
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


def _set_first_supported_arg(
    target: dict[str, Any],
    signature_params: dict[str, Any],
    candidate_names: tuple[str, ...],
    value: Any,
) -> str | None:
    for name in candidate_names:
        if name in signature_params:
            target[name] = value
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a security-focused LoRA adapter on the local refusal dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    args = parser.parse_args()

    train_config = load_yaml(args.config)
    project_config = load_yaml(Path(train_config["project_config"]))
    root = Path(project_config["project"]["root_dir"]).resolve()

    logging_dir = root / train_config["training"]["logging_dir"]
    log_path = logging_dir / "train.log"
    logger = setup_logger(log_path)

    try:
        set_global_seed(int(train_config["training"]["seed"]))
        model_name = project_config["model"]["base_model_id"]
        logger.info("Loading tokenizer: %s", model_name)
        tokenizer = load_tokenizer(model_name, trust_remote_code=project_config["model"]["trust_remote_code"])

        logger.info("Loading base model for LoRA training")
        model = load_causal_lm(
            model_name,
            trust_remote_code=project_config["model"]["trust_remote_code"],
            load_in_4bit=project_config["model"]["load_in_4bit"],
            use_bf16=train_config["training"]["bf16"],
            gradient_checkpointing=train_config["training"]["gradient_checkpointing"],
            device_preference=project_config["model"].get("device"),
            dtype_name=project_config["model"].get("dtype"),
            use_mps=bool(project_config["model"].get("use_mps", False)),
        )
        runtime_device = next(model.parameters()).device
        logger.info("Resolved runtime device: %s", runtime_device)
        if runtime_device.type != "cuda" and "8bit" in str(train_config["training"]["optim"]).lower():
            raise ValueError("8bit optimizers are CUDA-centric. Use adamw_torch or another non-8bit optimizer for MPS/CPU.")

        preferred_targets = list(train_config["lora"]["target_modules"])
        target_modules = infer_target_modules(model, preferred_targets)
        logger.info("Using target modules: %s", ", ".join(target_modules))
        model = attach_lora_adapter(model, train_config["lora"], target_modules)
        if bool(train_config["training"].get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        use_cache = bool(train_config["training"].get("use_cache", False))
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = use_cache
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = use_cache
        logger.info(
            "Memory settings: gradient_checkpointing=%s, use_cache=%s",
            bool(train_config["training"].get("gradient_checkpointing", False)),
            use_cache,
        )
        model.print_trainable_parameters()

        train_dataset = build_text_dataset(root / train_config["training"]["train_file"], tokenizer)
        eval_dataset = build_text_dataset(root / train_config["training"]["valid_file"], tokenizer)
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty.")

        checkpoint_dir = root / train_config["training"]["checkpoint_dir"]
        adapter_dir = root / train_config["training"]["output_adapter_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        training_signature = signature(TrainingArguments.__init__).parameters
        evaluation_strategy = str(train_config["training"].get("evaluation_strategy", "steps"))
        save_strategy = str(train_config["training"].get("save_strategy", "steps"))
        logging_strategy = str(train_config["training"].get("logging_strategy", "steps"))
        training_kwargs: dict[str, Any] = {
            "output_dir": str(checkpoint_dir),
            "num_train_epochs": float(train_config["training"]["num_train_epochs"]),
            "per_device_train_batch_size": int(train_config["training"]["per_device_train_batch_size"]),
            "per_device_eval_batch_size": int(train_config["training"]["per_device_eval_batch_size"]),
            "gradient_accumulation_steps": int(train_config["training"]["gradient_accumulation_steps"]),
            "learning_rate": float(train_config["training"]["learning_rate"]),
            "lr_scheduler_type": str(train_config["training"]["lr_scheduler_type"]),
            "warmup_ratio": float(train_config["training"]["warmup_ratio"]),
            "weight_decay": float(train_config["training"]["weight_decay"]),
            "logging_steps": int(train_config["training"]["logging_steps"]),
            "save_steps": int(train_config["training"]["save_steps"]),
            "eval_steps": int(train_config["training"]["eval_steps"]),
            "save_total_limit": int(train_config["training"]["save_total_limit"]),
            "bf16": bool(train_config["training"]["bf16"]),
            "max_steps": int(train_config["training"]["max_steps"]),
            "report_to": [],
            "remove_unused_columns": False,
            "optim": str(train_config["training"]["optim"]),
            "seed": int(train_config["training"]["seed"]),
        }
        training_kwargs = {
            key: value
            for key, value in training_kwargs.items()
            if key in training_signature
        }

        eval_strategy_key = _set_first_supported_arg(
            training_kwargs,
            training_signature,
            ("evaluation_strategy", "eval_strategy"),
            evaluation_strategy,
        )
        _set_first_supported_arg(training_kwargs, training_signature, ("save_strategy",), save_strategy)
        _set_first_supported_arg(training_kwargs, training_signature, ("logging_strategy",), logging_strategy)

        if "do_eval" in training_signature:
            training_kwargs["do_eval"] = len(eval_dataset) > 0 and evaluation_strategy.lower() != "no"
        if eval_strategy_key is None or evaluation_strategy.lower() == "no":
            training_kwargs.pop("eval_steps", None)
        if save_strategy.lower() == "no":
            training_kwargs.pop("save_steps", None)
        if logging_strategy.lower() != "steps":
            training_kwargs.pop("logging_steps", None)
        if "dataloader_num_workers" in training_signature:
            training_kwargs["dataloader_num_workers"] = int(train_config["training"].get("dataloader_num_workers", 0))
        if "dataloader_pin_memory" in training_signature:
            training_kwargs["dataloader_pin_memory"] = bool(train_config["training"].get("dataloader_pin_memory", runtime_device.type == "cuda"))
        if "dataloader_persistent_workers" in training_signature:
            training_kwargs["dataloader_persistent_workers"] = bool(
                train_config["training"].get("dataloader_persistent_workers", False)
            )
        if "skip_memory_metrics" in training_signature:
            training_kwargs["skip_memory_metrics"] = bool(train_config["training"].get("skip_memory_metrics", True))
        if "use_mps_device" in training_signature and "use_mps_device" in train_config["training"]:
            training_kwargs["use_mps_device"] = bool(train_config["training"]["use_mps_device"])
        training_args = TrainingArguments(**training_kwargs)

        trainer_signature = signature(SFTTrainer.__init__).parameters
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "args": training_args,
        }
        if "tokenizer" in trainer_signature:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = tokenizer
        if "dataset_text_field" in trainer_signature:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in trainer_signature:
            trainer_kwargs["max_seq_length"] = int(train_config["training"]["max_seq_length"])
        if "packing" in trainer_signature:
            trainer_kwargs["packing"] = False
        trainer = SFTTrainer(**trainer_kwargs)

        if runtime_device.type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
            gc.collect()
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache before training start")

        logger.info("Starting training")
        train_result = trainer.train()
        eval_metrics: dict[str, Any] = trainer.evaluate()

        logger.info("Saving adapter to %s", adapter_dir)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        save_json(logging_dir / "loss_history.json", trainer.state.log_history)
        save_json(logging_dir / "eval_during_train.json", eval_metrics)
        save_json(
            logging_dir / "train_summary.json",
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "target_modules": target_modules,
                "adapter_dir": str(adapter_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "runtime_device": str(runtime_device),
            },
        )
        logger.info("Training complete")
    except Exception as exc:  # pragma: no cover - defensive logging for long-running jobs
        logger.error("Training failed: %s", exc)
        logger.error("%s", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
