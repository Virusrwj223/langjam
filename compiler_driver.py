# compiler_driver.py
from __future__ import annotations

import json
import py_compile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gptlang_parse import parse_gptlang
from gptlang_validate import validate_main_v1, validate_game_v1

# ---------- LLM hooks (you will implement with LangChain) ----------

def llm_diagnostics(source_text: str, diagnostic_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a structured explanation + patches.
    Replace body with LangChain call.
    """
    return {
        "summary": "Validation failed.",
        "errors": diagnostic_json["diagnostics"],
        "patched_source": None,
    }

def llm_codegen(stage: str, source_text: str, validated_info: Dict[str, Any]) -> str:
    """
    Return Python code as a string.
    Replace body with LangChain call.
    """
    raise NotImplementedError

def llm_repair(py_code: str, py_error: str) -> str:
    """
    Return repaired Python code as a string.
    Replace body with LangChain call.
    """
    raise NotImplementedError

# ---------- Python compile check ----------

def python_compile_check(code: str) -> Tuple[bool, Optional[str]]:
    try:
        compile(code, "<generated>", "exec")
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}:{e.offset}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# ---------- Stage A compile ----------

def compile_main(main_path: Path, out_path: Path) -> int:
    src = main_path.read_text(encoding="utf-8")

    pr = parse_gptlang(src, source_name=str(main_path))
    if not pr.ok or pr.file_ast is None:
        print(json.dumps({
            "ok": False,
            "stage": "parse",
            "file": str(main_path),
            "diagnostics": [d.__dict__ for d in pr.diagnostics],  # or format like validate JSON
        }, indent=2))
        return 1

    vj = validate_main_v1(pr.file_ast, source_name=str(main_path))
    if not vj["ok"]:
        expl = llm_diagnostics(src, vj)
        print(json.dumps(expl, indent=2))
        return 1

    # valid → codegen
    py = llm_codegen(stage="meta", source_text=src, validated_info=vj)

    ok, err = python_compile_check(py)
    attempts = 0
    while not ok and attempts < 2:
        py = llm_repair(py, err or "unknown error")
        ok, err = python_compile_check(py)
        attempts += 1

    if not ok:
        print(json.dumps({
            "ok": False,
            "stage": "py_compile",
            "file": str(out_path),
            "diagnostics": [{"code": "E3001", "severity": "error", "message": err, "span": None}],
        }, indent=2))
        return 1

    out_path.write_text(py, encoding="utf-8")
    return 0

# ---------- Stage B compile ----------

def compile_game(game_path: Path, schema_path: Path, out_path: Path) -> int:
    src = game_path.read_text(encoding="utf-8")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    pr = parse_gptlang(src, source_name=str(game_path))
    if not pr.ok or pr.file_ast is None:
        print(json.dumps({
            "ok": False,
            "stage": "parse",
            "file": str(game_path),
            "diagnostics": [d.__dict__ for d in pr.diagnostics],
        }, indent=2))
        return 1

    vj = validate_game_v1(pr.file_ast, session_schema=schema, source_name=str(game_path))
    if not vj["ok"]:
        expl = llm_diagnostics(src, vj)
        print(json.dumps(expl, indent=2))
        return 1

    py = llm_codegen(stage="game", source_text=src, validated_info=vj)

    ok, err = python_compile_check(py)
    attempts = 0
    while not ok and attempts < 2:
        py = llm_repair(py, err or "unknown error")
        ok, err = python_compile_check(py)
        attempts += 1

    if not ok:
        print(json.dumps({
            "ok": False,
            "stage": "py_compile",
            "file": str(out_path),
            "diagnostics": [{"code": "E3001", "severity": "error", "message": err, "span": None}],
        }, indent=2))
        return 1

    out_path.write_text(py, encoding="utf-8")
    return 0
