from __future__ import annotations

import json
import subprocess
from typing import Any

import requests


class OllamaRunner:
    def __init__(self, *, model: str, mode: str, endpoint: str, timeout_seconds: int = 120) -> None:
        self.model = model
        self.mode = mode
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        system_prompt = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        user_prompt = "\n\n".join(message["content"] for message in messages if message["role"] == "user")
        if self.mode == "cli":
            combined_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
            result = subprocess.run(
                ["ollama", "run", self.model, combined_prompt],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout_seconds,
            )
            return result.stdout.strip()

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": 8192,
            },
        }
        if not do_sample:
            payload["options"]["temperature"] = temperature
        response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response: {json.dumps(data, ensure_ascii=False)}")
        return str(data["response"]).strip()
