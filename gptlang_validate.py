# gptlang_validate.py
# Deterministic validator for GPTLANG v1.
# Consumes Program AST (from gptlang_parse) and checks:
#   - required blocks present
#   - required keys present
#   - type checks (ValueKind)
#   - enum checks
#   - int range checks
#   - non-empty checks (string/list)
#   - list item kind + item enum checks
#   - conditional requirements (loop => tick_ms/max_turns)
#   - simple compatibility constraints (render=pygame policy, no_network policy if present)
#
# Produces CompilationResult with ok + diagnostics (+ ast if ok).

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from gptlang_ast import (
    Program,
    Block,
    KeyValue,
    Value,
    ValueKind,
    Diagnostic,
    Severity,
    CompilationResult,
    Err,
    Schema,
    GPTLANG_V1_SCHEMA,
    BlockSpec,
    FieldSpec,
    Range,
    field_spec_map,
    expected_fields,
)


def _span_of_block_or_program(block: Optional[Block]) -> Optional[object]:
    if block is None:
        return None
    return block.header_span


def _diag(
    code: str,
    message: str,
    *,
    span=None,
    context=None,
    suggestions=None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=Severity.ERROR,
        message=message,
        span=span,
        context=context or {},
        suggestions=suggestions or [],
    )


def validate_program(program: Program, *, schema: Schema = GPTLANG_V1_SCHEMA) -> CompilationResult:
    diags: List[Diagnostic] = []
    block_specs: Dict[str, BlockSpec] = {b.name: b for b in schema.blocks}

    # -------------------------
    # 1) Required blocks
    # -------------------------
    required_blocks = [b.name for b in schema.blocks if b.required]
    for bname in required_blocks:
        if not program.has_block(bname):
            diags.append(
                _diag(
                    Err.E1000_MISSING_BLOCK,
                    f"Missing required block @{bname}.",
                    span=None,
                    context={"required_blocks": required_blocks},
                    suggestions=[f"Add a @{bname} block with keys: {', '.join(expected_fields(block_specs[bname]))}."],
                )
            )

    # If blocks are missing, still continue to collect more errors where possible.

    # -------------------------
    # 2) Keys + type checks per block
    # -------------------------
    for bspec in schema.blocks:
        block = program.block(bspec.name)
        if block is None:
            continue

        fs_map = field_spec_map(bspec)

        # Unknown keys (if not allowed by spec) — this is a validation-time policy
        if not bspec.allow_extra_keys:
            for key in block.kvs.keys():
                if key not in fs_map:
                    kv = block.kvs[key]
                    diags.append(
                        _diag(
                            Err.E0001_UNEXPECTED,
                            f"Unknown key '{key}' in @{bspec.name}.",
                            span=kv.span or block.header_span,
                            context={"block": bspec.name, "known_keys": list(fs_map.keys())},
                            suggestions=["Fix the key name, or move it into the correct block."],
                        )
                    )

        # Missing required keys
        for fs in bspec.fields:
            if fs.required and not block.has(fs.key):
                diags.append(
                    _diag(
                        Err.E1001_MISSING_KEY,
                        f"Missing required key '{fs.key}' in @{bspec.name}.",
                        span=block.header_span,
                        context={"block": bspec.name, "required_keys": [f.key for f in bspec.fields if f.required]},
                        suggestions=[f"Add: {fs.key}: { _example_value(fs) }"],
                    )
                )

        # Validate each present key that is in schema
        for key, kv in block.kvs.items():
            fs = fs_map.get(key)
            if fs is None:
                continue  # already flagged as unknown key if policy enabled

            _validate_field(block_name=bspec.name, kv=kv, fs=fs, diags=diags)

    # -------------------------
    # 3) Cross-field conditional rules (v1)
    # -------------------------
    # RULES.tick_ms required if MECHANICS.loop == real_time
    # RULES.max_turns required if MECHANICS.loop == turn_based
    mechanics = program.block("MECHANICS")
    rules = program.block("RULES")

    loop_mode = None
    if mechanics is not None:
        v = mechanics.get("loop")
        if v is not None and v.kind == ValueKind.IDENT:
            loop_mode = v.value  # type: ignore[assignment]

    if loop_mode is not None:
        if rules is None:
            # RULES required anyway; missing handled above. Don't double-report.
            pass
        else:
            if loop_mode == "real_time":
                if not rules.has("tick_ms"):
                    diags.append(
                        _diag(
                            Err.E2000_MISSING_CONDITIONAL,
                            "Missing required key 'tick_ms' in @RULES because MECHANICS.loop is real_time.",
                            span=rules.header_span,
                            context={"loop": "real_time"},
                            suggestions=["Add: tick_ms: 100  # (16..1000)"],
                        )
                    )
            if loop_mode == "turn_based":
                if not rules.has("max_turns"):
                    diags.append(
                        _diag(
                            Err.E2000_MISSING_CONDITIONAL,
                            "Missing required key 'max_turns' in @RULES because MECHANICS.loop is turn_based.",
                            span=rules.header_span,
                            context={"loop": "turn_based"},
                            suggestions=["Add: max_turns: 200  # (1..10000)"],
                        )
                    )

    # OUTPUT.render policy (optional strictness)
    output = program.block("OUTPUT")
    if output is not None:
        render = output.get("render")
        if render is not None and render.kind == ValueKind.IDENT:
            if render.value == "pygame":
                # Hackathon default: allow but warn would be nicer; v1 uses errors only
                # If you *do* support pygame generation, delete this block.
                diags.append(
                    _diag(
                        Err.E2001_INCOMPATIBLE,
                        "render: pygame is not supported in v1 codegen (use ascii).",
                        span=(output.kvs.get("render").span if "render" in output.kvs else output.header_span),
                        context={"render": "pygame"},
                        suggestions=["Change to: render: ascii"],
                    )
                )

    # CONSTRAINTS.no_network policy (optional)
    constraints = program.block("CONSTRAINTS")
    if constraints is not None:
        nn = constraints.get("no_network")
        if nn is not None and nn.kind == ValueKind.BOOL:
            if nn.value is not True:
                diags.append(
                    _diag(
                        Err.E2002_UNSAFE,
                        "CONSTRAINTS.no_network must be true for this demo pipeline.",
                        span=(constraints.kvs.get("no_network").span if "no_network" in constraints.kvs else constraints.header_span),
                        suggestions=["Set: no_network: true"],
                    )
                )

    ok = len([d for d in diags if d.severity == Severity.ERROR]) == 0
    return CompilationResult(ok=ok, version=schema.version, errors=diags, ast=program if ok else None)


# -------------------------
# Field validation helpers
# -------------------------


def _validate_field(*, block_name: str, kv: KeyValue, fs: FieldSpec, diags: List[Diagnostic]) -> None:
    v = kv.value

    # Type check
    if v.kind != fs.kind:
        diags.append(
            _diag(
                Err.E1100_WRONG_TYPE,
                f"Wrong type for '{fs.key}' in @{block_name}: expected {fs.kind.value}, got {v.kind.value}.",
                span=v.span or kv.span,
                context={"block": block_name, "key": fs.key, "expected": fs.kind.value, "got": v.kind.value},
                suggestions=[f"Use: {fs.key}: { _example_value(fs) }"],
            )
        )
        return

    # Non-empty checks
    if fs.non_empty:
        if v.kind == ValueKind.STRING:
            if isinstance(v.value, str) and v.value.strip() == "":
                diags.append(
                    _diag(
                        Err.E1300_OUT_OF_RANGE,
                        f"'{fs.key}' in @{block_name} must be a non-empty string.",
                        span=v.span or kv.span,
                        suggestions=[f"Use: {fs.key}: \"...\""],
                    )
                )
                return
        if v.kind == ValueKind.LIST:
            items = v.value  # type: ignore[assignment]
            if isinstance(items, list) and len(items) == 0:
                diags.append(
                    _diag(
                        Err.E1300_OUT_OF_RANGE,
                        f"'{fs.key}' in @{block_name} must be a non-empty list.",
                        span=v.span or kv.span,
                        suggestions=[f"Use: {fs.key}: [{_example_list_item(fs)}]"],
                    )
                )
                return

    # Enum checks (IDENT)
    if v.kind == ValueKind.IDENT and fs.enum is not None:
        if v.value not in fs.enum:
            diags.append(
                _diag(
                    Err.E1200_INVALID_ENUM,
                    f"Invalid value for '{fs.key}' in @{block_name}: {v.value!r} is not allowed.",
                    span=v.span or kv.span,
                    context={"allowed": fs.enum},
                    suggestions=[f"Use one of: {', '.join(fs.enum)}"],
                )
            )
            return

    # Int range checks
    if v.kind == ValueKind.INT and fs.int_range is not None:
        try:
            iv = int(v.value)
        except Exception:
            diags.append(
                _diag(
                    Err.E1100_WRONG_TYPE,
                    f"'{fs.key}' in @{block_name} must be an int.",
                    span=v.span or kv.span,
                )
            )
            return
        if not fs.int_range.contains(iv):
            diags.append(
                _diag(
                    Err.E1300_OUT_OF_RANGE,
                    f"Out of range for '{fs.key}' in @{block_name}: {iv} not in { _range_str(fs.int_range) }.",
                    span=v.span or kv.span,
                    context={"min": fs.int_range.min, "max": fs.int_range.max},
                    suggestions=[f"Use: {fs.key}: { _example_int_in_range(fs.int_range) }"],
                )
            )
            return

    # List item checks
    if v.kind == ValueKind.LIST and fs.list_item_kind is not None:
        items = v.value  # type: ignore[assignment]
        if not isinstance(items, list):
            # should not happen if parser is correct
            diags.append(
                _diag(
                    Err.E1100_WRONG_TYPE,
                    f"'{fs.key}' in @{block_name} must be a list.",
                    span=v.span or kv.span,
                )
            )
            return

        for item in items:
            if not isinstance(item, Value):
                continue
            if item.kind != fs.list_item_kind:
                diags.append(
                    _diag(
                        Err.E1100_WRONG_TYPE,
                        f"Wrong item type in list '{fs.key}' in @{block_name}: expected {fs.list_item_kind.value}, got {item.kind.value}.",
                        span=item.span or v.span or kv.span,
                        context={"expected": fs.list_item_kind.value, "got": item.kind.value},
                        suggestions=[f"Example: {fs.key}: [{_example_list_item(fs)}]"],
                    )
                )
                return

            if fs.list_item_kind == ValueKind.IDENT and fs.list_item_enum is not None:
                if item.value not in fs.list_item_enum:
                    diags.append(
                        _diag(
                            Err.E1200_INVALID_ENUM,
                            f"Invalid action in '{fs.key}' in @{block_name}: {item.value!r} is not allowed.",
                            span=item.span or v.span or kv.span,
                            context={"allowed": fs.list_item_enum},
                            suggestions=[f"Use only: {', '.join(fs.list_item_enum)}"],
                        )
                    )
                    return


# -------------------------
# Example formatting helpers
# -------------------------


def _range_str(r: Range) -> str:
    lo = "-inf" if r.min is None else str(r.min)
    hi = "+inf" if r.max is None else str(r.max)
    return f"[{lo}..{hi}]"


def _example_int_in_range(r: Range) -> int:
    if r.min is not None:
        return r.min
    if r.max is not None:
        return r.max
    return 0


def _example_list_item(fs: FieldSpec) -> str:
    if fs.list_item_enum:
        return fs.list_item_enum[0]
    if fs.list_item_kind == ValueKind.IDENT:
        return "move"
    if fs.list_item_kind == ValueKind.STRING:
        return "\"x\""
    if fs.list_item_kind == ValueKind.INT:
        return "0"
    if fs.list_item_kind == ValueKind.BOOL:
        return "true"
    return "item"


def _example_value(fs: FieldSpec) -> str:
    if fs.kind == ValueKind.STRING:
        return "\"...\""
    if fs.kind == ValueKind.INT:
        if fs.int_range is not None:
            return str(_example_int_in_range(fs.int_range))
        return "0"
    if fs.kind == ValueKind.BOOL:
        return "true"
    if fs.kind == ValueKind.IDENT:
        if fs.enum:
            return fs.enum[0]
        return "ident_value"
    if fs.kind == ValueKind.LIST:
        item = _example_list_item(fs)
        return f"[{item}]"
    if fs.kind == ValueKind.MAP:
        return "{ key: \"value\" }"
    return "value"
