# openrouter_client.py
# Minimal OpenRouter chat client (stdlib + requests).
# Install: pip install requests

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from dotenv import load_dotenv
load_dotenv()



@dataclass
class OpenRouterConfig:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    # Optional but recommended by OpenRouter:
    http_referer: Optional[str] = None
    x_title: Optional[str] = None
    timeout_s: int = 60


class OpenRouterError(RuntimeError):
    pass


def chat_completion(
    *,
    cfg: OpenRouterConfig,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2500,
) -> str:
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    if cfg.http_referer:
        headers["HTTP-Referer"] = cfg.http_referer
    if cfg.x_title:
        headers["X-Title"] = cfg.x_title

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(cfg.base_url, headers=headers, data=json.dumps(payload), timeout=cfg.timeout_s)
    except requests.RequestException as e:
        raise OpenRouterError(f"OpenRouter request failed: {e}") from e

    content_type = resp.headers.get("Content-Type", "")
    text = resp.text or ""

    if resp.status_code < 200 or resp.status_code >= 300:
        snippet = text[:400].replace("\n", "\\n")
        raise OpenRouterError(
            f"OpenRouter HTTP {resp.status_code}. Content-Type={content_type!r}. Body(snippet)={snippet!r}"
        )

    try:
        data = resp.json()
    except Exception:
        snippet = text[:400].replace("\n", "\\n")
        raise OpenRouterError(
            "OpenRouter returned a non-JSON response.\n"
            f"base_url={cfg.base_url!r}\n"
            f"HTTP {resp.status_code}, Content-Type={content_type!r}\n"
            f"Body(snippet)={snippet!r}\n"
            "Likely causes:\n"
            "- OPENROUTER_BASE_URL is wrong (must be .../api/v1/chat/completions)\n"
            "- A proxy/captive portal returned HTML\n"
            "- Network/VPN interception\n"
        )

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise OpenRouterError(f"Unexpected OpenRouter response shape: {data}") from e



# openrouter_client.py (UPDATED)

def config_from_env(
    *,
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
) -> OpenRouterConfig:
    provider = os.environ.get("LLM_PROVIDER", "").strip()
    if provider != "openrouter":
        raise OpenRouterError(f"Unsupported LLM_PROVIDER: {provider!r}. Expected 'openrouter'.")

    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise OpenRouterError("Missing LLM_MODEL env var.")

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise OpenRouterError("Missing OPENROUTER_API_KEY env var.")

    base_url = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    if not base_url:
        raise OpenRouterError("Missing OPENROUTER_BASE_URL env var.")

    return OpenRouterConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        http_referer=http_referer,
        x_title=x_title,
    )

