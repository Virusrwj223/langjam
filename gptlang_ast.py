# gptlang_ast.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union


# -------------------------
# Source locations / spans
# -------------------------

@dataclass(frozen=True)
class Position:
    """1-based line/column position."""
    line: int
    col: int
    index: int  # 0-based absolute char offset


@dataclass(frozen=True)
class Span:
    start: Position
    end: Position  # exclusive


# -------------------------
# Keyword normalization
# -------------------------

class CanonSection(str, Enum):
    INTENT = "INTENT"
    WORLD = "WORLD"
    MECHANICS = "MECHANICS"
    RULES = "RULES"
    OUTPUT = "OUTPUT"
    CONSTRAINTS = "CONSTRAINTS"
    METADATA = "METADATA"
    PROGRAM = "PROGRAM"
    UNKNOWN = "UNKNOWN"  # for non-reserved custom sections


class CanonKey(str, Enum):
    # INTENT
    NAME = "name"
    VERSION = "version"
    STAGE = "stage"

    # WORLD
    SETTING = "setting"
    ASSETS = "assets"

    # MECHANICS
    TURNS = "turns"
    ACTIONS = "actions"
    WIN_CONDITION = "win_condition"

    # OUTPUT
    DSL_NAME = "dsl_name"
    EMIT_TEMPLATES = "emit_templates"

    # PROGRAM (Stage B)
    DSL = "dsl"
    STRATEGY = "strategy"

    UNKNOWN = "UNKNOWN"  # for custom keys (allowed; validator decides strictness)


@dataclass(frozen=True)
class NameRef:
    """
    Holds both raw surface lexeme (Gen-Z keyword) and canonical meaning
    used by validator/codegen.
    """
    raw: str
    canon: str  # CanonSection.value or CanonKey.value or "UNKNOWN"


# -------------------------
# Core AST
# -------------------------

@dataclass(frozen=True)
class File:
    sections: List["Section"]
    source_name: Optional[str] = None


@dataclass(frozen=True)
class Section:
    name: NameRef              # raw + canonical
    body: List["Statement"]
    span: Span


class StmtKind(str, Enum):
    FIELD = "field"
    BLOCK = "block"


@dataclass(frozen=True)
class Statement:
    kind: StmtKind
    value: Union["FieldStmt", "BlockStmt"]
    span: Span


@dataclass(frozen=True)
class FieldStmt:
    key: NameRef               # raw + canonical
    value: "Value"
    span: Span


@dataclass(frozen=True)
class BlockStmt:
    key: NameRef               # raw + canonical
    fields: List[FieldStmt]
    span: Span


# -------------------------
# Values (unchanged)
# -------------------------

class ValueKind(str, Enum):
    STRING = "string"
    INT = "int"
    BOOL = "bool"
    IDENT = "ident"
    LIST = "list"
    OBJECT = "object"


@dataclass(frozen=True)
class Value:
    kind: ValueKind
    value: Union["VString", "VInt", "VBool", "VIdent", "VList", "VObject"]
    span: Span


@dataclass(frozen=True)
class VString:
    text: str


@dataclass(frozen=True)
class VInt:
    n: int


@dataclass(frozen=True)
class VBool:
    b: bool


@dataclass(frozen=True)
class VIdent:
    name: str


@dataclass(frozen=True)
class VList:
    items: List[Value]


@dataclass(frozen=True)
class VObject:
    props: Dict[str, Value]


# -------------------------
# Convenience constructors
# -------------------------

def make_field(key: NameRef, val: Value, span: Span) -> Statement:
    fs = FieldStmt(key=key, value=val, span=span)
    return Statement(kind=StmtKind.FIELD, value=fs, span=span)


def make_block(key: NameRef, fields: List[FieldStmt], span: Span) -> Statement:
    bs = BlockStmt(key=key, fields=fields, span=span)
    return Statement(kind=StmtKind.BLOCK, value=bs, span=span)
