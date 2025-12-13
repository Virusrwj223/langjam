# compile.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from llm_client import LLMClient, LLMConfig
from llm_modes import llm_codegen, llm_diagnostics, llm_repair
from gptlang_parse import parse_gptlang
from gptlang_validate import validate_main_v1, validate_game_v1

def make_llm_client() -> tuple[LLMClient, str]:
    provider = os.getenv("LLM_PROVIDER", "openrouter")
    model = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    if not provider or not model:
        raise RuntimeError("Missing LLM_PROVIDER or LLM_MODEL in .env")

    if provider == "openrouter":
        base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        client = LLMClient(LLMConfig(base_url=base, api_key_env="OPENROUTER_API_KEY", default_model=model))
        return client, model

    if provider == "groq":
        base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        client = LLMClient(LLMConfig(base_url=base, api_key_env="GROQ_API_KEY", default_model=model))
        return client, model

    raise RuntimeError(f"Unknown LLM_PROVIDER={provider}")

def python_compile_check(code: str) -> tuple[bool, str | None]:
    try:
        compile(code, "<generated>", "exec")
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def compile_main(main_llm: Path, out_py: Path) -> int:
    src = main_llm.read_text(encoding="utf-8")
    pr = parse_gptlang(src, source_name=str(main_llm))
    if not pr.ok or pr.file_ast is None:
        # parse diagnostics are dataclasses; keep it simple for now
        print(json.dumps({
            "ok": False,
            "stage": "parse",
            "file": str(main_llm),
            "diagnostics": [d.__dict__ for d in pr.diagnostics],
        }, indent=2))
        return 1

    vj = validate_main_v1(pr.file_ast, source_name=str(main_llm))
    if not vj["ok"]:
        client, _ = make_llm_client()
        expl = llm_diagnostics(client, src, vj)
        print(json.dumps(expl, indent=2))
        return 1

    client, _ = make_llm_client()
    py = llm_codegen(client, stage="meta", source_text=src, validated_info=vj)

    ok, err = python_compile_check(py)
    attempts = 0
    while not ok and attempts < 2:
        py = llm_repair(client, py, err or "unknown error")
        ok, err = python_compile_check(py)
        attempts += 1

    if not ok:
        print(json.dumps({
            "ok": False,
            "stage": "py_compile",
            "file": str(out_py),
            "diagnostics": [{"code": "E3001", "severity": "error", "message": err, "span": None}],
        }, indent=2))
        return 1

    out_py.write_text(py, encoding="utf-8")
    print(json.dumps({"ok": True, "stage": "emit", "file": str(out_py)}, indent=2))
    return 0

def compile_game(game_llm: Path, schema_json: Path, out_py: Path) -> int:
    src = game_llm.read_text(encoding="utf-8")
    schema = json.loads(schema_json.read_text(encoding="utf-8"))

    pr = parse_gptlang(src, source_name=str(game_llm))
    if not pr.ok or pr.file_ast is None:
        print(json.dumps({
            "ok": False,
            "stage": "parse",
            "file": str(game_llm),
            "diagnostics": [d.__dict__ for d in pr.diagnostics],
        }, indent=2))
        return 1

    vj = validate_game_v1(pr.file_ast, session_schema=schema, source_name=str(game_llm))
    if not vj["ok"]:
        client, _ = make_llm_client()
        expl = llm_diagnostics(client, src, vj)
        print(json.dumps(expl, indent=2))
        return 1

    client, _ = make_llm_client()
    py = llm_codegen(client, stage="game", source_text=src, validated_info=vj)

    ok, err = python_compile_check(py)
    attempts = 0
    while not ok and attempts < 2:
        py = llm_repair(client, py, err or "unknown error")
        ok, err = python_compile_check(py)
        attempts += 1

    if not ok:
        print(json.dumps({
            "ok": False,
            "stage": "py_compile",
            "file": str(out_py),
            "diagnostics": [{"code": "E3001", "severity": "error", "message": err, "span": None}],
        }, indent=2))
        return 1

    out_py.write_text(py, encoding="utf-8")
    print(json.dumps({"ok": True, "stage": "emit", "file": str(out_py)}, indent=2))
    return 0

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("meta")
    a.add_argument("main_llm")
    a.add_argument("--out", default="gen_game.py")

    b = sub.add_parser("game")
    b.add_argument("game_llm")
    b.add_argument("--schema", default="session_schema.json")
    b.add_argument("--out", default="play_game.py")

    args = ap.parse_args()
    if args.cmd == "meta":
        raise SystemExit(compile_main(Path(args.main_llm), Path(args.out)))
    else:
        raise SystemExit(compile_game(Path(args.game_llm), Path(args.schema), Path(args.out)))

if __name__ == "__main__":
    main()
