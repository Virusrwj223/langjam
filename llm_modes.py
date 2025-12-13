# llm_modes.py
from __future__ import annotations
import json
from typing import Any, Dict, List

from llm_client import LLMClient

DIAGNOSTICS_SYSTEM = """You are a compiler diagnostics engine.
You will be given:
1) the user's .llm source text
2) machine validator diagnostics JSON (authoritative)
Task:
- Explain errors in compiler style
- Provide minimal suggested patches
Output MUST be valid JSON with keys: summary, errors, patched_source (nullable).
Do NOT invent errors not present in diagnostics JSON.
"""

CODEGEN_SYSTEM = """You are a compiler backend that generates Python code from a validated .llm spec.
Rules:
- Generate a complete, runnable single Python file.
- No placeholders, no TODOs.
- Determinism: follow the validated spec exactly.
- Stage meta: write session_schema.json and print templates.
- Stage game: generate play_game.py logic accordingly.
Output ONLY the Python code, no markdown, no commentary.
All files written by generated Python code MUST be written relative to the
directory containing the generated file:
BASE_DIR = Path(__file__).resolve().parent
Never write files relative to the current working directory.
Do not print the DSL/templates to stdout. Write them to dsl_doc.txt in the same directory as the generated file.
"""

REPAIR_SYSTEM = """You are a compiler repair backend.
You will be given Python code and a Python compile error.
Fix the code with minimal changes.
Output ONLY the corrected full Python code, no markdown, no commentary.
"""

def llm_diagnostics(client: LLMClient, source_text: str, diagnostic_json: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": DIAGNOSTICS_SYSTEM},
        {"role": "user", "content": json.dumps({
            "source_text": source_text,
            "diagnostics": diagnostic_json,
        })},
    ]
    out = client.chat(messages)
    return json.loads(out)  # if this fails, treat as toolchain error

def llm_codegen(client: LLMClient, stage: str, source_text: str, validated_info: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": CODEGEN_SYSTEM},
        {"role": "user", "content": json.dumps({
            "stage": stage,
            "source_text": source_text,
            "validated_info": validated_info,
        })},
    ]
    return client.chat(messages)

def llm_repair(client: LLMClient, py_code: str, py_error: str) -> str:
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": json.dumps({
            "python_code": py_code,
            "python_error": py_error,
        })},
    ]
    return client.chat(messages)
