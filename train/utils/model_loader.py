from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_bnb_config(load_in_4bit: bool = True, use_bf16: bool = True) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_causal_lm(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = True,
    load_in_4bit: bool = True,
    use_bf16: bool = True,
    gradient_checkpointing: bool = False,
) -> Any:
    quantization_config = build_bnb_config(load_in_4bit=load_in_4bit, use_bf16=use_bf16)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def infer_target_modules(model: Any, preferred_targets: list[str]) -> list[str]:
    module_names = [name for name, _ in model.named_modules()]
    available_suffixes = {name.rsplit(".", 1)[-1] for name in module_names}
    matched = [name for name in preferred_targets if name in available_suffixes]
    if matched:
        return matched

    suffix_counter: Counter[str] = Counter()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if "linear" in class_name:
            suffix_counter[name.rsplit(".", 1)[-1]] += 1
    return [suffix for suffix, _ in suffix_counter.most_common(8)]


def attach_lora_adapter(model: Any, lora_config: dict[str, Any], target_modules: list[str]) -> Any:
    prepared_model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
        target_modules=target_modules,
    )
    return get_peft_model(prepared_model, peft_config)


def load_model_with_adapter(
    model_name_or_path: str,
    adapter_path: str,
    *,
    trust_remote_code: bool = True,
    load_in_4bit: bool = True,
    use_bf16: bool = True,
) -> Any:
    base_model = load_causal_lm(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        use_bf16=use_bf16,
        gradient_checkpointing=False,
    )
    return PeftModel.from_pretrained(base_model, adapter_path)
