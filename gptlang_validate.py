# gptlang_validate.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from gptlang_ast import (
    BlockStmt,
    CanonKey,
    CanonSection,
    FieldStmt,
    File,
    Section,
    Span,
    Statement,
    StmtKind,
    Value,
    ValueKind,
    VBool,
    VIdent,
    VInt,
    VList,
    VObject,
    VString,
)

# ============================================================
# 0) Diagnostic JSON helpers
# ============================================================

def span_to_json(span: Span) -> Dict[str, int]:
    return {
        "line": span.start.line,
        "col": span.start.col,
        "endLine": span.end.line,
        "endCol": span.end.col,
        "index": span.start.index,
        "endIndex": span.end.index,
    }


def diag_json(
    code: str,
    message: str,
    span: Span,
    severity: str = "error",
    context: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "span": span_to_json(span),
        "context": context or {},
        "suggestions": suggestions or [],
    }


def result_json(
    ok: bool,
    stage: str,
    file: str,
    diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "ok": ok,
        "stage": stage,
        "file": file,
        "diagnostics": diagnostics,
    }


# ============================================================
# 1) AST indexing helpers
# ============================================================

def index_sections(ast: File) -> Dict[str, Section]:
    # Map by canonical section name. If multiple unknown/custom sections appear,
    # only last wins; duplicates should already be parse errors, but we stay safe.
    out: Dict[str, Section] = {}
    for s in ast.sections:
        out[s.name.canon] = s
    return out


def index_fields(sec: Section) -> Dict[str, FieldStmt]:
    out: Dict[str, FieldStmt] = {}
    for st in sec.body:
        if st.kind == StmtKind.FIELD:
            fs = st.value  # type: ignore[assignment]
            assert isinstance(fs, FieldStmt)
            out[fs.key.canon] = fs
    return out


def index_blocks(sec: Section) -> Dict[str, BlockStmt]:
    out: Dict[str, BlockStmt] = {}
    for st in sec.body:
        if st.kind == StmtKind.BLOCK:
            bs = st.value  # type: ignore[assignment]
            assert isinstance(bs, BlockStmt)
            out[bs.key.canon] = bs
    return out


def file_span(ast: File) -> Span:
    # Best-effort: span from first to last section
    if not ast.sections:
        # dummy span (caller should never validate an empty parsed file in practice)
        from gptlang_ast import Position
        p = Position(1, 1, 0)
        return Span(p, p)
    return Span(ast.sections[0].span.start, ast.sections[-1].span.end)


# ============================================================
# 2) Value checks (type, enum, range, non-empty list, etc.)
# ============================================================

def is_ident_value(v: Value) -> bool:
    return v.kind == ValueKind.IDENT

def is_string_value(v: Value) -> bool:
    return v.kind == ValueKind.STRING

def is_int_value(v: Value) -> bool:
    return v.kind == ValueKind.INT

def is_bool_value(v: Value) -> bool:
    return v.kind == ValueKind.BOOL

def is_list_value(v: Value) -> bool:
    return v.kind == ValueKind.LIST

def get_ident(v: Value) -> Optional[str]:
    if v.kind == ValueKind.IDENT:
        return v.value.name  # type: ignore[attr-defined]
    return None

def get_string(v: Value) -> Optional[str]:
    if v.kind == ValueKind.STRING:
        return v.value.text  # type: ignore[attr-defined]
    return None

def get_int(v: Value) -> Optional[int]:
    if v.kind == ValueKind.INT:
        return v.value.n  # type: ignore[attr-defined]
    return None

def get_bool(v: Value) -> Optional[bool]:
    if v.kind == ValueKind.BOOL:
        return v.value.b  # type: ignore[attr-defined]
    return None

def get_list(v: Value) -> Optional[List[Value]]:
    if v.kind == ValueKind.LIST:
        return v.value.items  # type: ignore[attr-defined]
    return None


def expect_type(
    diags: List[Dict[str, Any]],
    v: Value,
    expected: str,
) -> bool:
    """
    expected in {"string","int","bool","ident","list","object"}
    """
    kind_map = {
        "string": ValueKind.STRING,
        "int": ValueKind.INT,
        "bool": ValueKind.BOOL,
        "ident": ValueKind.IDENT,
        "list": ValueKind.LIST,
        "object": ValueKind.OBJECT,
    }
    ek = kind_map[expected]
    if v.kind != ek:
        diags.append(diag_json(
            "E1003",
            f"Type mismatch: expected {expected}, found {v.kind.value}",
            v.span,
            context={"expected": expected, "found": v.kind.value},
        ))
        return False
    return True


def expect_enum_ident(
    diags: List[Dict[str, Any]],
    v: Value,
    allowed: List[str],
) -> bool:
    if not expect_type(diags, v, "ident"):
        return False
    val = get_ident(v)
    if val not in allowed:
        diags.append(diag_json(
            "E1004",
            f"Invalid enum value: {val!r} (allowed: {allowed})",
            v.span,
            context={"allowed": allowed, "found": val},
        ))
        return False
    return True


def expect_int_range(
    diags: List[Dict[str, Any]],
    v: Value,
    lo: int,
    hi: int,
) -> bool:
    if not expect_type(diags, v, "int"):
        return False
    n = get_int(v)
    assert n is not None
    if n < lo or n > hi:
        diags.append(diag_json(
            "E1005",
            f"Integer out of range: {n} not in [{lo}, {hi}]",
            v.span,
            context={"min": lo, "max": hi, "found": n},
        ))
        return False
    return True


def expect_nonempty_list_of_idents(
    diags: List[Dict[str, Any]],
    v: Value,
) -> Optional[List[str]]:
    if not expect_type(diags, v, "list"):
        return None
    items = get_list(v) or []
    if len(items) == 0:
        diags.append(diag_json(
            "E1006",
            "List must be non-empty",
            v.span,
        ))
        return None

    out: List[str] = []
    for it in items:
        if it.kind != ValueKind.IDENT:
            diags.append(diag_json(
                "E1003",
                f"List element type mismatch: expected ident, found {it.kind.value}",
                it.span,
                context={"expected_elem": "ident", "found_elem": it.kind.value},
            ))
            continue
        out.append(it.value.name)  # type: ignore[attr-defined]
    return out


# ============================================================
# 3) validate_main_v1(ast)
# ============================================================

def validate_main_v1(ast: File, source_name: str = "main.llm") -> Dict[str, Any]:
    diags: List[Dict[str, Any]] = []
    secs = index_sections(ast)

    # ---- required sections (canonical) ----
    required_sections = [
        CanonSection.INTENT.value,
        CanonSection.WORLD.value,
        CanonSection.MECHANICS.value,
        CanonSection.OUTPUT.value,
    ]
    for sec_name in required_sections:
        if sec_name not in secs:
            diags.append(diag_json(
                "E1002",
                f"Missing required section @{sec_name}",
                file_span(ast),
                context={"expected": required_sections, "found": [s.name.canon for s in ast.sections]},
                suggestions=[{
                    "title": f"Add @{sec_name} section template",
                    "patch": _main_section_template(sec_name),
                }],
            ))

    # If critical sections missing, we still continue to report more, but
    # some checks depend on them existing.
    # ---- INTENT ----
    intent = secs.get(CanonSection.INTENT.value)
    if intent:
        fields = index_fields(intent)

        _require_field(diags, intent, fields, CanonKey.NAME.value, expected="string")
        _require_field(diags, intent, fields, CanonKey.VERSION.value, expected="int")
        _require_field(diags, intent, fields, CanonKey.STAGE.value, expected="ident")

        # version must be 1
        if CanonKey.VERSION.value in fields:
            v = fields[CanonKey.VERSION.value].value
            if expect_type(diags, v, "int"):
                n = get_int(v)
                if n != 1:
                    diags.append(diag_json(
                        "E1004",
                        f"Invalid version: expected 1, found {n}",
                        v.span,
                        context={"expected": 1, "found": n},
                        suggestions=[{"title": "Set patch/version to 1", "patch": "patch: 1;"}],
                    ))

        # stage must be meta
        if CanonKey.STAGE.value in fields:
            v = fields[CanonKey.STAGE.value].value
            expect_enum_ident(diags, v, allowed=["meta"])

    # ---- WORLD ----
    world = secs.get(CanonSection.WORLD.value)
    assets: List[str] = []
    if world:
        fields = index_fields(world)
        _require_field(diags, world, fields, CanonKey.SETTING.value, expected="string")
        _require_field(diags, world, fields, CanonKey.ASSETS.value, expected="list")

        if CanonKey.ASSETS.value in fields:
            v = fields[CanonKey.ASSETS.value].value
            got = expect_nonempty_list_of_idents(diags, v)
            if got is not None:
                assets = got

    # ---- MECHANICS ----
    mech = secs.get(CanonSection.MECHANICS.value)
    if mech:
        fields = index_fields(mech)
        _require_field(diags, mech, fields, CanonKey.TURNS.value, expected="int")
        _require_field(diags, mech, fields, CanonKey.ACTIONS.value, expected="list")
        _require_field(diags, mech, fields, CanonKey.WIN_CONDITION.value, expected="string")

        if CanonKey.TURNS.value in fields:
            expect_int_range(diags, fields[CanonKey.TURNS.value].value, lo=1, hi=10_000)

        if CanonKey.ACTIONS.value in fields:
            actions = expect_nonempty_list_of_idents(diags, fields[CanonKey.ACTIONS.value].value)
            # optional: actions must be unique
            if actions:
                dupes = sorted({a for a in actions if actions.count(a) > 1})
                if dupes:
                    diags.append(diag_json(
                        "E1004",
                        f"Duplicate actions not allowed: {dupes}",
                        fields[CanonKey.ACTIONS.value].value.span,
                        context={"duplicates": dupes},
                    ))

    # ---- OUTPUT ----
    out = secs.get(CanonSection.OUTPUT.value)
    if out:
        fields = index_fields(out)
        _require_field(diags, out, fields, CanonKey.DSL_NAME.value, expected="ident")
        _require_field(diags, out, fields, CanonKey.EMIT_TEMPLATES.value, expected="bool")

    ok = not any(d["severity"] == "error" for d in diags)
    return result_json(ok=ok, stage="validate", file=source_name, diagnostics=diags)


def _require_field(
    diags: List[Dict[str, Any]],
    sec: Section,
    fields: Dict[str, FieldStmt],
    key_canon: str,
    expected: Optional[str] = None,
) -> None:
    if key_canon not in fields:
        diags.append(diag_json(
            "E1001",
            f"Missing required field '{key_canon}' in section @{sec.name.raw}",
            sec.span,
            context={"section": sec.name.raw, "missing": key_canon},
            suggestions=[{
                "title": f"Add field '{key_canon}'",
                "patch": _field_patch_example(sec.name.canon, key_canon),
            }],
        ))
        return
    if expected is not None:
        expect_type(diags, fields[key_canon].value, expected)


def _main_section_template(sec_canon: str) -> str:
    # canonical templates; slang mapping happens at the surface, but patch can be canonical too.
    if sec_canon == CanonSection.INTENT.value:
        return "@VIBE_CHECK {\n  title: \"...\";\n  patch: 1;\n  mode: meta;\n}\n"
    if sec_canon == CanonSection.WORLD.value:
        return "@WORLD_BUILD {\n  setting: \"...\";\n  loot: [ASSET1];\n}\n"
    if sec_canon == CanonSection.MECHANICS.value:
        return "@HOW_IT_HITS {\n  turns: 100;\n  moves: [move1];\n  dub_condition: \"...\";\n}\n"
    if sec_canon == CanonSection.OUTPUT.value:
        return "@SHIP_IT {\n  session_dsl: SessionDSL;\n  print_templates: true;\n}\n"
    return f"@{sec_canon} {{\n}}\n"


def _field_patch_example(section_canon: str, key_canon: str) -> str:
    # Prefer slang patches for user-facing experience.
    # Minimal v1 cases:
    if section_canon == CanonSection.INTENT.value:
        if key_canon == CanonKey.NAME.value:
            return 'title: "My Game";'
        if key_canon == CanonKey.VERSION.value:
            return "patch: 1;"
        if key_canon == CanonKey.STAGE.value:
            return "mode: meta;"
    if section_canon == CanonSection.WORLD.value:
        if key_canon == CanonKey.SETTING.value:
            return 'setting: "neon city";'
        if key_canon == CanonKey.ASSETS.value:
            return "loot: [BTC];"
    if section_canon == CanonSection.MECHANICS.value:
        if key_canon == CanonKey.TURNS.value:
            return "turns: 50;"
        if key_canon == CanonKey.ACTIONS.value:
            return "moves: [trade];"
        if key_canon == CanonKey.WIN_CONDITION.value:
            return 'dub_condition: "profit max";'
    if section_canon == CanonSection.OUTPUT.value:
        if key_canon == CanonKey.DSL_NAME.value:
            return "session_dsl: MySessionDSL;"
        if key_canon == CanonKey.EMIT_TEMPLATES.value:
            return "print_templates: true;"
    # fallback
    return f"{key_canon}: ...;"


# ============================================================
# 4) validate_game_v1(ast, session_schema)
# ============================================================

# A realistic minimal schema you can generate from Stage A:
# session_schema = {
#   "dsl_name": "VibeExchangeSession",
#   "program": {
#       "required_block": "strategy",           # canonical
#       "block_schema": {
#           "required": {
#               "asset": {"type": "ident", "in": ["BTC", "ETH"]},
#               "risk":  {"type": "ident", "enum": ["low", "med", "high"]},
#           },
#           "optional": {
#               "rule":  {"type": "string"},
#           },
#           "strict_unknown_keys": False
#       }
#   }
# }

def validate_game_v1(
    ast: File,
    session_schema: Dict[str, Any],
    source_name: str = "game1.llm",
) -> Dict[str, Any]:
    diags: List[Dict[str, Any]] = []
    secs = index_sections(ast)

    # Required: PROGRAM section (slang: @PLAYBOOK)
    prog = secs.get(CanonSection.PROGRAM.value)
    if not prog:
        diags.append(diag_json(
            "E1002",
            "Missing required section @PROGRAM (slang: @PLAYBOOK)",
            file_span(ast),
            suggestions=[{
                "title": "Add @PLAYBOOK template",
                "patch": _game_playbook_template(session_schema),
            }],
        ))
        ok = False
        return result_json(ok=ok, stage="validate", file=source_name, diagnostics=diags)

    # Validate PROGRAM fields
    prog_fields = index_fields(prog)
    prog_blocks = index_blocks(prog)

    # 1) dsl must match schema["dsl_name"]
    expected_dsl = session_schema.get("dsl_name")
    if CanonKey.DSL.value not in prog_fields:
        diags.append(diag_json(
            "E1001",
            f"Missing required field '{CanonKey.DSL.value}' in @PLAYBOOK",
            prog.span,
            suggestions=[{"title": "Set session_dsl", "patch": f"session_dsl: {expected_dsl};"}] if expected_dsl else [],
        ))
    else:
        v = prog_fields[CanonKey.DSL.value].value
        if expect_type(diags, v, "ident"):
            got = get_ident(v)
            if expected_dsl is not None and got != expected_dsl:
                diags.append(diag_json(
                    "E2002",
                    f"DSL mismatch: expected {expected_dsl}, found {got}",
                    v.span,
                    context={"expected": expected_dsl, "found": got},
                    suggestions=[{"title": "Fix DSL name", "patch": f"session_dsl: {expected_dsl};"}],
                ))

    # 2) required block (canonical "strategy") exists (slang: gameplan)
    program_spec = session_schema.get("program", {})
    required_block_canon = program_spec.get("required_block", CanonKey.STRATEGY.value)

    if required_block_canon not in prog_blocks:
        diags.append(diag_json(
            "E1001",
            f"Missing required block '{required_block_canon}' in @PLAYBOOK (slang: gameplan)",
            prog.span,
            suggestions=[{"title": "Add gameplan block", "patch": _gameplan_template(session_schema)}],
        ))
        ok = not any(d["severity"] == "error" for d in diags)
        return result_json(ok=ok, stage="validate", file=source_name, diagnostics=diags)

    strategy = prog_blocks[required_block_canon]
    block_schema = program_spec.get("block_schema", {})
    _validate_block_against_schema(diags, strategy, block_schema)

    ok = not any(d["severity"] == "error" for d in diags)
    return result_json(ok=ok, stage="validate", file=source_name, diagnostics=diags)


def _validate_block_against_schema(
    diags: List[Dict[str, Any]],
    block: BlockStmt,
    schema: Dict[str, Any],
) -> None:
    """
    schema shape:
    {
      "required": { key: { "type": "...", "enum": [...], "in": [...] , "min":..,"max":..,"nonempty":.. } },
      "optional": { ... },
      "strict_unknown_keys": bool
    }
    """
    required: Dict[str, Any] = schema.get("required", {})
    optional: Dict[str, Any] = schema.get("optional", {})
    strict_unknown = bool(schema.get("strict_unknown_keys", False))

    # index by canonical key (already canon_key-mapped by parser)
    fields_by_canon: Dict[str, FieldStmt] = {f.key.canon: f for f in block.fields}

    # required keys present
    for key_canon, spec in required.items():
        if key_canon not in fields_by_canon:
            diags.append(diag_json(
                "E1001",
                f"Missing required field '{key_canon}' in block {block.key.raw}",
                block.span,
                context={"block": block.key.raw, "missing": key_canon},
                suggestions=[{"title": f"Add '{key_canon}'", "patch": _schema_field_patch(key_canon, spec)}],
            ))
        else:
            _validate_value_against_spec(diags, fields_by_canon[key_canon].value, spec)

    # optional keys: if present, validate
    for key_canon, spec in optional.items():
        if key_canon in fields_by_canon:
            _validate_value_against_spec(diags, fields_by_canon[key_canon].value, spec)

    # unknown keys (if strict)
    if strict_unknown:
        allowed = set(required.keys()) | set(optional.keys())
        for k in fields_by_canon.keys():
            if k not in allowed:
                diags.append(diag_json(
                    "E1007",
                    f"Unknown key '{k}' in block {block.key.raw}",
                    fields_by_canon[k].span,
                    severity="error",
                    context={"allowed": sorted(list(allowed)), "found": k},
                ))


def _validate_value_against_spec(
    diags: List[Dict[str, Any]],
    v: Value,
    spec: Dict[str, Any],
) -> None:
    ty = spec.get("type")
    if ty:
        if not expect_type(diags, v, ty):
            return

    # enum for ident
    if "enum" in spec:
        allowed = list(spec["enum"])
        expect_enum_ident(diags, v, allowed=allowed)

    # membership constraint (for ident/string typically)
    if "in" in spec:
        allowed = list(spec["in"])
        if v.kind == ValueKind.IDENT:
            got = get_ident(v)
        elif v.kind == ValueKind.STRING:
            got = get_string(v)
        else:
            got = None

        if got is None or got not in allowed:
            diags.append(diag_json(
                "E2001",
                f"Value {got!r} not allowed (expected one of {allowed})",
                v.span,
                context={"allowed": allowed, "found": got},
            ))

    # numeric range
    if v.kind == ValueKind.INT and ("min" in spec or "max" in spec):
        lo = int(spec.get("min", -2**31))
        hi = int(spec.get("max", 2**31 - 1))
        expect_int_range(diags, v, lo=lo, hi=hi)

    # list constraints
    if v.kind == ValueKind.LIST and spec.get("nonempty") is True:
        items = get_list(v) or []
        if len(items) == 0:
            diags.append(diag_json("E1006", "List must be non-empty", v.span))


def _schema_field_patch(key_canon: str, spec: Dict[str, Any]) -> str:
    ty = spec.get("type", "ident")
    if ty == "ident":
        if "enum" in spec and spec["enum"]:
            return f"{key_canon}: {spec['enum'][0]};"
        if "in" in spec and spec["in"]:
            return f"{key_canon}: {spec['in'][0]};"
        return f"{key_canon}: SOME_IDENT;"
    if ty == "string":
        return f'{key_canon}: "...";'
    if ty == "int":
        lo = spec.get("min", 0)
        return f"{key_canon}: {lo};"
    if ty == "bool":
        return f"{key_canon}: true;"
    if ty == "list":
        return f"{key_canon}: [ITEM];"
    return f"{key_canon}: ...;"


def _game_playbook_template(session_schema: Dict[str, Any]) -> str:
    dsl = session_schema.get("dsl_name", "SessionDSL")
    return (
        "@PLAYBOOK {\n"
        f"  session_dsl: {dsl};\n"
        "  gameplan {\n"
        "    # fill according to template printed by gen_game.py\n"
        "  }\n"
        "}\n"
    )


def _gameplan_template(session_schema: Dict[str, Any]) -> str:
    program_spec = session_schema.get("program", {})
    block_schema = program_spec.get("block_schema", {})
    required: Dict[str, Any] = block_schema.get("required", {})
    lines = ["gameplan {"]
    for k, spec in required.items():
        lines.append(f"  {_schema_field_patch(k, spec)}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# # ============================================================
# # 5) Minimal usage example
# # ============================================================

# if __name__ == "__main__":
#     from gptlang_parse import parse_gptlang

#     main_src = r"""
#     @VIBE_CHECK { title: "X"; patch: 1; mode: meta; }
#     @WORLD_BUILD { setting: "Y"; loot: [BTC, ETH]; }
#     @HOW_IT_HITS { turns: 50; moves: [trade]; dub_condition: "profit"; }
#     @SHIP_IT { session_dsl: VibeExchangeSession; print_templates: true; }
#     """
#     pr = parse_gptlang(main_src, "main.llm")
#     assert pr.ok and pr.file_ast
#     print(validate_main_v1(pr.file_ast, "main.llm"))

#     game_src = r"""
#     @PLAYBOOK {
#       session_dsl: VibeExchangeSession;
#       gameplan {
#         asset: BTC;
#         risk: low;
#       }
#     }
#     """
#     pr2 = parse_gptlang(game_src, "game1.llm")
#     assert pr2.ok and pr2.file_ast
#     schema = {
#         "dsl_name": "VibeExchangeSession",
#         "program": {
#             "required_block": CanonKey.STRATEGY.value,
#             "block_schema": {
#                 "required": {
#                     "asset": {"type": "ident", "in": ["BTC", "ETH"]},
#                     "risk": {"type": "ident", "enum": ["low", "med", "high"]},
#                 },
#                 "optional": {
#                     "rule": {"type": "string"},
#                 },
#                 "strict_unknown_keys": False,
#             },
#         },
#     }
#     print(validate_game_v1(pr2.file_ast, schema, "game1.llm"))
