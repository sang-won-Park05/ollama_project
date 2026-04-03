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
    ) -> None:
        self.tokenizer = load_tokenizer(model_name_or_path, trust_remote_code=trust_remote_code)
        if adapter_path:
            self.model = load_model_with_adapter(
                model_name_or_path,
                adapter_path,
                trust_remote_code=trust_remote_code,
                load_in_4bit=load_in_4bit,
                use_bf16=use_bf16,
            )
        else:
            self.model = load_causal_lm(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                load_in_4bit=load_in_4bit,
                use_bf16=use_bf16,
                gradient_checkpointing=False,
            )
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        prompt = apply_chat_template_to_messages(self.tokenizer, messages, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        generated = outputs[0][prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
