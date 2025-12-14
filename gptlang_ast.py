# gptlang_ast.py
# GPTLANG v1 — minimal AST + diagnostics model (deterministic compiler frontend)
# This file is intentionally self-contained and stdlib-only.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# -----------------------------
# Source locations (for errors)
# -----------------------------

@dataclass(frozen=True)
class Span:
    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "start_line": self.start_line,
            "start_col": self.start_col,
            "end_line": self.end_line,
            "end_col": self.end_col,
        }


# -----------------------------
# Diagnostics (compiler errors)
# -----------------------------

class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"  # v1: typically unused, but reserved


@dataclass
class Diagnostic:
    code: str
    severity: Severity
    message: str
    span: Optional[Span] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "span": None if self.span is None else self.span.to_dict(),
            "context": self.context,
            "suggestions": self.suggestions,
        }


@dataclass
class CompilationResult:
    ok: bool
    version: str = "gptlang-v1"
    errors: List[Diagnostic] = field(default_factory=list)
    ast: Optional["Program"] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "errors": [e.to_dict() for e in self.errors],
            "ast": None if self.ast is None else self.ast.to_dict(),
        }


# -----------------------------
# Value AST (deterministic DSL)
# -----------------------------

class ValueKind(str, Enum):
    STRING = "string"
    INT = "int"
    BOOL = "bool"
    IDENT = "ident"
    LIST = "list"
    MAP = "map"


@dataclass(frozen=True)
class Value:
    kind: ValueKind
    value: Any
    span: Optional[Span] = None

    def to_python(self) -> Any:
        return self.value

    def _json_value(self) -> Any:
        # Recursively convert nested Values in LIST/MAP
        if self.kind == ValueKind.LIST:
            return [v.to_dict() if isinstance(v, Value) else v for v in (self.value or [])]
        if self.kind == ValueKind.MAP:
            return {k: (v.to_dict() if isinstance(v, Value) else v) for k, v in (self.value or {}).items()}
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "value": self._json_value(),
            "span": None if self.span is None else self.span.to_dict(),
        }



def VString(s: str, span: Optional[Span] = None) -> Value:
    return Value(ValueKind.STRING, s, span)


def VInt(i: int, span: Optional[Span] = None) -> Value:
    return Value(ValueKind.INT, i, span)


def VBool(b: bool, span: Optional[Span] = None) -> Value:
    return Value(ValueKind.BOOL, b, span)


def VIdent(name: str, span: Optional[Span] = None) -> Value:
    return Value(ValueKind.IDENT, name, span)


def VList(items: List[Value], span: Optional[Span] = None) -> Value:
    return Value(ValueKind.LIST, items, span)


def VMap(items: Dict[str, Value], span: Optional[Span] = None) -> Value:
    return Value(ValueKind.MAP, items, span)


# -----------------------------
# Program AST (blocks + kvs)
# -----------------------------

@dataclass
class KeyValue:
    key: str
    value: Value
    span: Optional[Span] = None  # span of whole kv line (optional)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value.to_dict(),
            "span": None if self.span is None else self.span.to_dict(),
        }


@dataclass
class Block:
    name: str  # e.g., "INTENT"
    kvs: Dict[str, KeyValue] = field(default_factory=dict)
    header_span: Optional[Span] = None

    def has(self, key: str) -> bool:
        return key in self.kvs

    def get(self, key: str) -> Optional[Value]:
        kv = self.kvs.get(key)
        return None if kv is None else kv.value

    def items(self) -> List[KeyValue]:
        return list(self.kvs.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "header_span": None if self.header_span is None else self.header_span.to_dict(),
            "kvs": {k: kv.to_dict() for k, kv in self.kvs.items()},
        }


@dataclass
class Program:
    blocks: Dict[str, Block] = field(default_factory=dict)

    def has_block(self, name: str) -> bool:
        return name in self.blocks

    def block(self, name: str) -> Optional[Block]:
        return self.blocks.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {"blocks": {k: b.to_dict() for k, b in self.blocks.items()}}


# -----------------------------
# Error codes (centralized)
# -----------------------------

class Err:
    # Syntax / structure
    E0001_UNEXPECTED = "E0001"     # Unexpected token / cannot parse line
    E0002_UNTERMINATED_STRING = "E0002"
    E0003_INVALID_COLLECTION = "E0003"  # list/map syntax
    E0100_DUP_BLOCK = "E0100"
    E0101_DUP_KEY = "E0101"
    E0102_UNKNOWN_BLOCK = "E0102"  # optional policy

    # Schema / typing
    E1000_MISSING_BLOCK = "E1000"
    E1001_MISSING_KEY = "E1001"
    E1100_WRONG_TYPE = "E1100"
    E1200_INVALID_ENUM = "E1200"
    E1300_OUT_OF_RANGE = "E1300"

    # Cross-field / conditional
    E2000_MISSING_CONDITIONAL = "E2000"
    E2001_INCOMPATIBLE = "E2001"
    E2002_UNSAFE = "E2002"


# -----------------------------
# Schema (v1) — used by validator
# -----------------------------

@dataclass(frozen=True)
class Range:
    min: Optional[int] = None
    max: Optional[int] = None

    def contains(self, x: int) -> bool:
        if self.min is not None and x < self.min:
            return False
        if self.max is not None and x > self.max:
            return False
        return True


@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: ValueKind
    required: bool = True
    enum: Optional[List[str]] = None        # for IDENT
    int_range: Optional[Range] = None       # for INT
    non_empty: bool = False                 # for STRING / LIST
    # for LIST typing, you can constrain items:
    list_item_kind: Optional[ValueKind] = None
    list_item_enum: Optional[List[str]] = None  # if item_kind == IDENT


@dataclass(frozen=True)
class BlockSpec:
    name: str
    required: bool = True
    fields: List[FieldSpec] = field(default_factory=list)
    allow_extra_keys: bool = False


@dataclass(frozen=True)
class Schema:
    version: str
    blocks: List[BlockSpec]
    allow_extra_blocks: bool = False


# -----------------------------
# GPTLANG v1 fixed schema
# -----------------------------

GPTLANG_V1_SCHEMA = Schema(
    version="gptlang-v1",
    allow_extra_blocks=False,
    blocks=[
        BlockSpec(
            name="INTENT",
            required=True,
            allow_extra_keys=False,
            fields=[
                FieldSpec("title", ValueKind.STRING, required=True, non_empty=True),
                FieldSpec("patch", ValueKind.INT, required=True, int_range=Range(1, None)),
                FieldSpec("mode", ValueKind.IDENT, required=True,
                          enum=["vibe_quest", "grindset_runner", "brainrot_arena"]),
                FieldSpec("audience_vibe", ValueKind.IDENT, required=True,
                          enum=["cozy", "sweaty", "chaotic", "cinematic"]),
            ],
        ),
        BlockSpec(
            name="WORLD",
            required=True,
            allow_extra_keys=False,
            fields=[
                FieldSpec("setting", ValueKind.STRING, required=True, non_empty=True),
                FieldSpec("tone", ValueKind.IDENT, required=True,
                          enum=["wholesome", "unhinged", "spooky", "chill", "absurd"]),
                FieldSpec("seed", ValueKind.INT, required=True, int_range=Range(0, 999_999)),
            ],
        ),
        BlockSpec(
            name="MECHANICS",
            required=True,
            allow_extra_keys=False,
            fields=[
                FieldSpec("genre", ValueKind.IDENT, required=True,
                          enum=["roguelike", "platformer", "narrative", "puzzle", "arcade"]),
                FieldSpec("loop", ValueKind.IDENT, required=True,
                          enum=["turn_based", "real_time"]),
                FieldSpec("player_hp", ValueKind.INT, required=True, int_range=Range(1, 999)),
                FieldSpec("win_condition", ValueKind.IDENT, required=True,
                          enum=["reach_goal", "survive_timer", "score_threshold"]),
                FieldSpec("lose_condition", ValueKind.IDENT, required=True,
                          enum=["hp_zero", "timeout"]),
            ],
        ),
        BlockSpec(
            name="RULES",
            required=True,
            allow_extra_keys=False,
            fields=[
                FieldSpec(
                    "actions",
                    ValueKind.LIST,
                    required=True,
                    non_empty=True,
                    list_item_kind=ValueKind.IDENT,
                    list_item_enum=["move", "jump", "dash", "attack", "interact", "wait"],
                ),
                # conditional keys enforced by validator (not required here)
                FieldSpec("tick_ms", ValueKind.INT, required=False, int_range=Range(16, 1000)),
                FieldSpec("max_turns", ValueKind.INT, required=False, int_range=Range(1, 10_000)),
                FieldSpec("difficulty", ValueKind.IDENT, required=True, enum=["ez", "normal", "cracked"]),
            ],
        ),
        BlockSpec(
            name="OUTPUT",
            required=True,
            allow_extra_keys=False,
            fields=[
                FieldSpec("target", ValueKind.IDENT, required=True, enum=["python"]),
                FieldSpec("entrypoint", ValueKind.IDENT, required=True, enum=["main"]),
                FieldSpec("render", ValueKind.IDENT, required=True, enum=["ascii", "pygame"]),
                FieldSpec("logging", ValueKind.BOOL, required=True),
            ],
        ),
        BlockSpec(
            name="CONSTRAINTS",
            required=False,
            allow_extra_keys=False,
            fields=[
                FieldSpec("no_network", ValueKind.BOOL, required=False),
                FieldSpec("max_loc", ValueKind.INT, required=False, int_range=Range(200, 2000)),
                FieldSpec("max_runtime_s", ValueKind.INT, required=False, int_range=Range(1, 30)),
            ],
        ),
    ],
)


# -----------------------------
# Small helpers for validator
# -----------------------------

def expected_fields(block_spec: BlockSpec) -> List[str]:
    return [f.key for f in block_spec.fields]


def field_spec_map(block_spec: BlockSpec) -> Dict[str, FieldSpec]:
    return {f.key: f for f in block_spec.fields}
