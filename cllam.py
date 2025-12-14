# gptlang_compile.py (UPDATED)

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

from gptlang_parse import parse_file
from gptlang_validate import validate_program
from gptlang_prompts import diagnostics_prompt, codegen_prompt
from openrouter_client import config_from_env, chat_completion, OpenRouterError


@dataclass
class CompileOptions:
    out_py_name: str = "gen_game.py"
    http_referer: Optional[str] = None
    x_title: Optional[str] = "GPTLANG Compiler"


def _print_diagnostics_json(diags) -> None:
    payload = {
        "ok": False,
        "version": "gptlang-v1",
        "errors": [d.to_dict() for d in diags],
    }
    sys.stderr.write(json.dumps(payload, indent=2) + "\n")


def _find_single_llm_file(dir_path: str) -> str:
    files = [f for f in os.listdir(dir_path) if f.endswith(".llm")]
    if len(files) == 0:
        raise RuntimeError("No .llm file found in directory.")
    if len(files) > 1:
        raise RuntimeError(f"Multiple .llm files found: {files}. Only one is allowed.")
    return os.path.join(dir_path, files[0])


def compile_project(project_dir: str, *, opts: CompileOptions) -> int:
    if not os.path.isdir(project_dir):
        sys.stderr.write(f"Not a directory: {project_dir}\n")
        return 2

    try:
        llm_path = _find_single_llm_file(project_dir)
    except RuntimeError as e:
        sys.stderr.write(str(e) + "\n")
        return 2

    out_py_path = os.path.join(project_dir, opts.out_py_name)

    # -------------------------
    # Phase 1: Parse
    # -------------------------
    parse_res = parse_file(llm_path)
    if not parse_res.ok:
        _print_diagnostics_json(parse_res.errors)
        _explain_with_gpt(parse_res.errors, opts)
        return 1

    # -------------------------
    # Phase 2: Validate
    # -------------------------
    val_res = validate_program(parse_res.ast)  # type: ignore[arg-type]
    if not val_res.ok:
        _print_diagnostics_json(val_res.errors)
        _explain_with_gpt(val_res.errors, opts)
        return 1

    # -------------------------
    # Phase 3: Codegen (GPT)
    # -------------------------
    try:
        cfg = config_from_env(http_referer=opts.http_referer, x_title=opts.x_title)
        py_code = chat_completion(
            cfg=cfg,
            messages=codegen_prompt(val_res.ast.to_dict()),  # type: ignore[union-attr]
            temperature=0.2,
            max_tokens=3000,
        )
    except OpenRouterError as e:
        sys.stderr.write(f"OpenRouter error: {e}\n")
        return 3

    if "def main" not in py_code or "__name__" not in py_code:
        sys.stderr.write("Codegen failed: missing main() or __name__ guard.\n")
        return 4

    with open(out_py_path, "w", encoding="utf-8") as f:
        f.write(py_code.rstrip() + "\n")

    print(f"OK: generated {out_py_path}")
    return 0


def _explain_with_gpt(errors, opts: CompileOptions) -> None:
    try:
        cfg = config_from_env(http_referer=opts.http_referer, x_title=opts.x_title)
        msg = chat_completion(
            cfg=cfg,
            messages=diagnostics_prompt([e.to_dict() for e in errors]),
            temperature=0.2,
            max_tokens=1200,
        )
        sys.stderr.write("\n=== GPT Diagnostics Explanation ===\n")
        sys.stderr.write(msg.strip() + "\n")
    except Exception:
        pass  # Diagnostics must never depend on GPT availability


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gptlangc",
        description="GPTLANG v1 compiler: <project_dir> -> <project_dir>/gen_game.py",
    )
    p.add_argument("project_dir", help="Directory containing exactly one .llm file")
    p.add_argument(
        "-o",
        "--out-name",
        default="main.py",
        help="Output Python filename (inside project dir)",
    )
    p.add_argument("--http-referer", default=None)
    p.add_argument("--x-title", default="GPTLANG Compiler")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    opts = CompileOptions(
        out_py_name=args.out_name,
        http_referer=args.http_referer,
        x_title=args.x_title,
    )
    return compile_project(args.project_dir, opts=opts)


if __name__ == "__main__":
    raise SystemExit(main())
