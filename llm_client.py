# llm_client.py
from __future__ import annotations
import os, json
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class LLMConfig:
    base_url: str          # e.g. "https://openrouter.ai/api/v1"
    api_key_env: str       # e.g. "OPENROUTER_API_KEY"
    default_model: str     # e.g. "meta-llama/llama-3.1-8b-instruct:free"
    timeout_s: int = 60

class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        key = os.getenv(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key in env var {cfg.api_key_env}")
        self.api_key = key

    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """
        OpenAI-style Chat Completions request.
        Works with OpenRouter and many compatible providers.
        """
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model or self.cfg.default_model,
            "messages": messages,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",  # can be any site/app identifier
        }

        # OpenRouter recommends additional headers for attribution (optional but good practice)
        # headers["HTTP-Referer"] = "http://localhost"
        # headers["X-Title"] = "gptlang-compiler"

        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout_s)
        if r.status_code >= 400:
            print("[LLM] status:", r.status_code)
            print("[LLM] body:", r.text[:1000])
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
