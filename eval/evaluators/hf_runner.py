from __future__ import annotations

from typing import Any

import torch

from train.utils import (
    apply_chat_template_to_messages,
    load_causal_lm,
    load_model_with_adapter,
    load_tokenizer,
)


class HFRunner:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        adapter_path: str | None,
        trust_remote_code: bool,
        load_in_4bit: bool,
        use_bf16: bool,
        device_preference: str | None = None,
        dtype_name: str | None = None,
        use_mps: bool = False,
    ) -> None:
        self.tokenizer = load_tokenizer(model_name_or_path, trust_remote_code=trust_remote_code)
        if adapter_path:
            self.model = load_model_with_adapter(
                model_name_or_path,
                adapter_path,
                trust_remote_code=trust_remote_code,
                load_in_4bit=load_in_4bit,
                use_bf16=use_bf16,
                device_preference=device_preference,
                dtype_name=dtype_name,
                use_mps=use_mps,
            )
        else:
            self.model = load_causal_lm(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                load_in_4bit=load_in_4bit,
                use_bf16=use_bf16,
                gradient_checkpointing=False,
                device_preference=device_preference,
                dtype_name=dtype_name,
                use_mps=use_mps,
            )
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "use_cache"):
            generation_config.use_cache = True
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        timeout_seconds: int | None = None,
        max_input_tokens: int | None = None,
    ) -> str:
        prompt = apply_chat_template_to_messages(self.tokenizer, messages, add_generation_prompt=True)
        tokenize_kwargs: dict[str, Any] = {"return_tensors": "pt"}
        if max_input_tokens is not None and max_input_tokens > 0:
            tokenize_kwargs["truncation"] = True
            tokenize_kwargs["max_length"] = int(max_input_tokens)
        encoded = self.tokenizer(prompt, **tokenize_kwargs)
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        if timeout_seconds is not None and timeout_seconds > 0:
            generation_kwargs["max_time"] = float(timeout_seconds)
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        generated = outputs[0][prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
