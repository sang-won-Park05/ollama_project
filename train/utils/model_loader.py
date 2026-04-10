from __future__ import annotations

from collections import Counter
from types import MethodType
from typing import Any

import torch
from torch import nn
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


def resolve_runtime_device(preferred_device: str | None = None, use_mps: bool = False) -> torch.device:
    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(runtime_device: torch.device, use_bf16: bool = True, dtype_name: str | None = None) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if runtime_device.type == "cuda":
        return torch.bfloat16 if use_bf16 else torch.float16
    if runtime_device.type == "mps":
        return torch.float16
    return torch.float32


def _is_exaone_model(model: Any) -> bool:
    class_name = model.__class__.__name__.lower()
    module_name = model.__class__.__module__.lower()
    config_type = model.config.model_type.lower() if getattr(model, "config", None) and getattr(model.config, "model_type", None) else ""
    return "exaone" in class_name or "exaone" in module_name or "exaone" in config_type


def _has_working_input_embeddings(module: Any) -> bool:
    if not hasattr(module, "get_input_embeddings"):
        return False
    try:
        embeddings = module.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        return False
    return isinstance(embeddings, nn.Module)


def _bind_input_embedding_methods(target: Any, getter_attr: str, setter_attr: str) -> None:
    def _getter(self: Any) -> nn.Module:
        embeddings = getattr(self, getter_attr, None)
        if embeddings is None:
            raise AttributeError(f"Missing embedding attribute: {getter_attr}")
        return embeddings

    def _setter(self: Any, value: nn.Module) -> None:
        setattr(self, setter_attr, value)

    target.get_input_embeddings = MethodType(_getter, target)
    target.set_input_embeddings = MethodType(_setter, target)


def ensure_input_embedding_accessors(model: Any) -> Any:
    if not _is_exaone_model(model):
        return model

    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "wte") and not _has_working_input_embeddings(transformer):
        _bind_input_embedding_methods(transformer, "wte", "wte")

    if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and not _has_working_input_embeddings(model):
        def _model_getter(self: Any) -> nn.Module:
            return self.transformer.wte

        def _model_setter(self: Any, value: nn.Module) -> None:
            self.transformer.wte = value

        model.get_input_embeddings = MethodType(_model_getter, model)
        model.set_input_embeddings = MethodType(_model_setter, model)

    return model


def build_bnb_config(
    runtime_device: torch.device,
    load_in_4bit: bool = True,
    use_bf16: bool = True,
) -> BitsAndBytesConfig | None:
    if not load_in_4bit or runtime_device.type != "cuda":
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
    device_preference: str | None = None,
    dtype_name: str | None = None,
    use_mps: bool = False,
) -> Any:
    runtime_device = resolve_runtime_device(preferred_device=device_preference, use_mps=use_mps)
    quantization_config = build_bnb_config(
        runtime_device=runtime_device,
        load_in_4bit=load_in_4bit,
        use_bf16=use_bf16,
    )
    torch_dtype = resolve_torch_dtype(runtime_device=runtime_device, use_bf16=use_bf16, dtype_name=dtype_name)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif runtime_device.type == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model = ensure_input_embedding_accessors(model)
    if runtime_device.type != "cuda":
        model.to(runtime_device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True
    setattr(model, "_ollama_project_runtime_device", runtime_device.type)
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
    model = ensure_input_embedding_accessors(model)
    prepared_model = model
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        prepared_model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

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
    device_preference: str | None = None,
    dtype_name: str | None = None,
    use_mps: bool = False,
) -> Any:
    base_model = load_causal_lm(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        use_bf16=use_bf16,
        gradient_checkpointing=False,
        device_preference=device_preference,
        dtype_name=dtype_name,
        use_mps=use_mps,
    )
    base_model = ensure_input_embedding_accessors(base_model)
    return PeftModel.from_pretrained(base_model, adapter_path)
