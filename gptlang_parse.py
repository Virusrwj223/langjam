# gptlang_parse.py
# Deterministic parser for GPTLANG v1 (sectioned DSL).
# - Line-based block parser: @BLOCK then key: value lines.
# - Small value parser for string/int/bool/ident/list/map.
# - Produces Program AST + structured diagnostics.
#
# This module does NOT do full schema validation (required keys, enums, ranges).
# It *does* enforce:
#   - block header syntax
#   - kv line syntax
#   - value syntax
#   - duplicate blocks
#   - duplicate keys within a block
#   - optional: forbid unknown blocks (if schema.allow_extra_blocks == False)
#   - optional: forbid unknown keys per block (if block_spec.allow_extra_keys == False)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from gptlang_ast import (
    Program,
    Block,
    KeyValue,
    Value,
    ValueKind,
    VString,
    VInt,
    VBool,
    VIdent,
    VList,
    VMap,
    Span,
    Diagnostic,
    Severity,
    CompilationResult,
    Err,
    Schema,
    GPTLANG_V1_SCHEMA,
    field_spec_map,
)

# -----------------------------
# Internal parsing cursor
# -----------------------------


@dataclass
class _Cursor:
    s: str
    i: int = 0

    def eof(self) -> bool:
        return self.i >= len(self.s)

    def peek(self) -> str:
        return "\0" if self.eof() else self.s[self.i]

    def take(self) -> str:
        ch = self.peek()
        if not self.eof():
            self.i += 1
        return ch

    def consume_ws(self) -> None:
        while not self.eof() and self.peek() in (" ", "\t", "\r", "\n"):
            self.i += 1

    def consume_inline_ws(self) -> None:
        while not self.eof() and self.peek() in (" ", "\t"):
            self.i += 1


# -----------------------------
# Lex helpers
# -----------------------------


def _strip_comment(line: str) -> str:
    # Comment delimiter is #, but ignore # inside strings would require full lexing.
    # v1 compromise: treat # as comment start only if it occurs outside quotes.
    in_str = False
    esc = False
    for idx, ch in enumerate(line):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if ch == "#" and not in_str:
            return line[:idx]
    return line


def _make_span_for_line(line_no: int, start_col: int, end_col: int) -> Span:
    return Span(start_line=line_no, start_col=start_col, end_line=line_no, end_col=end_col)


def _is_blank(line: str) -> bool:
    return line.strip() == ""


def _is_block_header(line: str) -> bool:
    line = line.strip()
    return line.startswith("@") and len(line) >= 2


def _parse_block_name(line: str) -> Optional[str]:
    # @BLOCK_NAME where BLOCK_NAME matches [A-Z][A-Z0-9_]*
    t = line.strip()
    if not t.startswith("@"):
        return None
    name = t[1:].strip()
    if name == "":
        return None
    if not ("A" <= name[0] <= "Z"):
        return None
    for ch in name:
        if not (("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch == "_"):
            return None
    return name


def _parse_key_value_line(line: str) -> Optional[Tuple[str, str, int]]:
    """
    Returns (key, raw_value, colon_index) if line contains a kv.
    Key matches [a-z][a-z0-9_]*
    """
    # We already stripped comments.
    if ":" not in line:
        return None
    colon = line.find(":")
    key = line[:colon].strip()
    raw = line[colon + 1 :].strip()

    if key == "":
        return None
    if not ("a" <= key[0] <= "z"):
        return None
    for ch in key:
        if not (("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_"):
            return None
    return key, raw, colon


# -----------------------------
# Value parser (deterministic)
# -----------------------------


def _parse_value(raw: str, line_no: int, start_col: int) -> Tuple[Optional[Value], List[Diagnostic]]:
    """
    Parse a value from raw string. start_col is the 1-based column where value starts.
    Returns (Value|None, diagnostics)
    """
    diags: List[Diagnostic] = []
    cur = _Cursor(raw, 0)

    def error(code: str, msg: str, col_start: int, col_end: int, suggestions: Optional[List[str]] = None) -> None:
        diags.append(
            Diagnostic(
                code=code,
                severity=Severity.ERROR,
                message=msg,
                span=_make_span_for_line(line_no, col_start, col_end),
                suggestions=suggestions or [],
            )
        )

    def parse_string() -> Optional[Value]:
        # assumes current char is "
        col0 = start_col + cur.i
        cur.take()  # opening quote
        out_chars: List[str] = []
        esc = False
        while not cur.eof():
            ch = cur.take()
            if esc:
                # minimal escapes
                if ch in ['"', "\\", "n", "t", "r"]:
                    if ch == "n":
                        out_chars.append("\n")
                    elif ch == "t":
                        out_chars.append("\t")
                    elif ch == "r":
                        out_chars.append("\r")
                    else:
                        out_chars.append(ch)
                else:
                    out_chars.append(ch)  # permissive
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                col1 = start_col + cur.i  # already advanced past closing "
                return VString("".join(out_chars), span=_make_span_for_line(line_no, col0, col1))
            if ch == "\n":
                break
            out_chars.append(ch)

        # Unterminated string
        col1 = start_col + cur.i
        error(
            Err.E0002_UNTERMINATED_STRING,
            "Unterminated string literal.",
            col0,
            max(col0, col1),
            suggestions=['Close the string with a trailing quote: "...".'],
        )
        return None

    def parse_int_or_ident_or_bool() -> Optional[Value]:
        col0 = start_col + cur.i
        # read token until whitespace or , ] }
        tok_chars: List[str] = []
        while not cur.eof():
            ch = cur.peek()
            if ch in (" ", "\t", "\r", "\n", ",", "]", "}", "{"):
                break
            tok_chars.append(cur.take())
        tok = "".join(tok_chars)
        col1 = start_col + cur.i

        if tok == "":
            error(Err.E0001_UNEXPECTED, "Expected a value.", col0, max(col0, col1))
            return None

        if tok == "true":
            return VBool(True, span=_make_span_for_line(line_no, col0, max(col0, col1)))
        if tok == "false":
            return VBool(False, span=_make_span_for_line(line_no, col0, max(col0, col1)))

        # int?
        if tok[0] == "-" or tok[0].isdigit():
            ok = True
            for j, ch in enumerate(tok):
                if j == 0 and ch == "-":
                    continue
                if not ch.isdigit():
                    ok = False
                    break
            if ok:
                try:
                    return VInt(int(tok), span=_make_span_for_line(line_no, col0, max(col0, col1)))
                except ValueError:
                    pass  # fallthrough

        # ident (A-Za-z_ then A-Za-z0-9_)
        if tok and (tok[0].isalpha() or tok[0] == "_"):
            for ch in tok[1:]:
                if not (ch.isalnum() or ch == "_"):
                    error(
                        Err.E0001_UNEXPECTED,
                        f"Invalid identifier token: {tok!r}.",
                        col0,
                        max(col0, col1),
                        suggestions=["Use only letters, digits, and underscores for identifiers."],
                    )
                    return None
            return VIdent(tok, span=_make_span_for_line(line_no, col0, max(col0, col1)))

        error(
            Err.E0001_UNEXPECTED,
            f"Could not parse value token: {tok!r}.",
            col0,
            max(col0, col1),
            suggestions=['Use a string "…", an int like 123, a bool true/false, or an identifier like foo_bar.'],
        )
        return None

    def parse_list() -> Optional[Value]:
        col0 = start_col + cur.i
        cur.take()  # '['
        cur.consume_inline_ws()
        items: List[Value] = []

        if not cur.eof() and cur.peek() == "]":
            cur.take()
            col1 = start_col + cur.i
            return VList(items, span=_make_span_for_line(line_no, col0, col1))

        while not cur.eof():
            cur.consume_inline_ws()
            item = parse_value_any()
            if item is None:
                return None
            items.append(item)
            cur.consume_inline_ws()
            if cur.eof():
                break
            if cur.peek() == ",":
                cur.take()
                continue
            if cur.peek() == "]":
                cur.take()
                col1 = start_col + cur.i
                return VList(items, span=_make_span_for_line(line_no, col0, col1))
            # unexpected
            colx = start_col + cur.i
            error(
                Err.E0003_INVALID_COLLECTION,
                "Invalid list syntax: expected ',' or ']'.",
                colx,
                colx,
                suggestions=["Lists look like: [move, attack, wait]"],
            )
            return None

        col1 = start_col + cur.i
        error(
            Err.E0003_INVALID_COLLECTION,
            "Unterminated list: missing closing ']'.",
            col0,
            max(col0, col1),
            suggestions=["Add a closing bracket: ]"],
        )
        return None

    def parse_map() -> Optional[Value]:
        col0 = start_col + cur.i
        cur.take()  # '{'
        cur.consume_inline_ws()
        items: Dict[str, Value] = {}

        if not cur.eof() and cur.peek() == "}":
            cur.take()
            col1 = start_col + cur.i
            return VMap(items, span=_make_span_for_line(line_no, col0, col1))

        while not cur.eof():
            cur.consume_inline_ws()
            # parse key
            k_col0 = start_col + cur.i
            key_chars: List[str] = []
            while not cur.eof():
                ch = cur.peek()
                if ch in (" ", "\t", ":", "\r", "\n", ",", "}"):
                    break
                key_chars.append(cur.take())
            k = "".join(key_chars).strip()
            k_col1 = start_col + cur.i

            if k == "":
                error(
                    Err.E0003_INVALID_COLLECTION,
                    "Invalid map syntax: expected a key.",
                    k_col0,
                    max(k_col0, k_col1),
                    suggestions=['Maps look like: { key: "value", other: 123 }'],
                )
                return None

            # key must match [a-z][a-z0-9_]*
            if not ("a" <= k[0] <= "z") or any(
                not (("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "_") for ch in k
            ):
                error(
                    Err.E0003_INVALID_COLLECTION,
                    f"Invalid map key {k!r}. Keys must be lowercase identifiers.",
                    k_col0,
                    max(k_col0, k_col1),
                    suggestions=["Use keys like: some_key: ..."],
                )
                return None

            cur.consume_inline_ws()
            if cur.eof() or cur.peek() != ":":
                colx = start_col + cur.i
                error(
                    Err.E0003_INVALID_COLLECTION,
                    "Invalid map syntax: expected ':' after key.",
                    colx,
                    colx,
                )
                return None
            cur.take()  # ':'
            cur.consume_inline_ws()

            v = parse_value_any()
            if v is None:
                return None
            items[k] = v

            cur.consume_inline_ws()
            if cur.eof():
                break
            if cur.peek() == ",":
                cur.take()
                continue
            if cur.peek() == "}":
                cur.take()
                col1 = start_col + cur.i
                return VMap(items, span=_make_span_for_line(line_no, col0, col1))

            colx = start_col + cur.i
            error(
                Err.E0003_INVALID_COLLECTION,
                "Invalid map syntax: expected ',' or '}'.",
                colx,
                colx,
            )
            return None

        col1 = start_col + cur.i
        error(
            Err.E0003_INVALID_COLLECTION,
            "Unterminated map: missing closing '}'.",
            col0,
            max(col0, col1),
            suggestions=["Add a closing brace: }"],
        )
        return None

    def parse_value_any() -> Optional[Value]:
        cur.consume_inline_ws()
        if cur.eof():
            col0 = start_col + cur.i
            error(Err.E0001_UNEXPECTED, "Expected a value.", col0, col0)
            return None
        ch = cur.peek()
        if ch == '"':
            return parse_string()
        if ch == "[":
            return parse_list()
        if ch == "{":
            return parse_map()
        return parse_int_or_ident_or_bool()

    v = parse_value_any()
    if v is None:
        return None, diags

    # allow trailing whitespace but no extra junk
    cur.consume_inline_ws()
    if not cur.eof():
        col0 = start_col + cur.i
        col1 = start_col + len(raw)
        diags.append(
            Diagnostic(
                code=Err.E0001_UNEXPECTED,
                severity=Severity.ERROR,
                message="Unexpected trailing characters after value.",
                span=_make_span_for_line(line_no, col0, max(col0, col1)),
                suggestions=["Remove extra characters after the value."],
            )
        )
        return None, diags

    return v, diags


# -----------------------------
# Main parse entrypoints
# -----------------------------


def parse_string(
    src: str,
    *,
    schema: Schema = GPTLANG_V1_SCHEMA,
    enforce_known_blocks: bool = True,
    enforce_known_keys: bool = False,
) -> CompilationResult:
    """
    Deterministic parse of GPTLANG file contents.

    - enforce_known_blocks:
        if True and schema.allow_extra_blocks == False, unknown @BLOCK is an error.
    - enforce_known_keys:
        if True, unknown keys in a known block are parse-time errors when block_spec.allow_extra_keys == False.
        (You can set this False and push all key checks to schema validation.)
    """
    program = Program()
    diags: List[Diagnostic] = []

    block_specs: Dict[str, object] = {b.name: b for b in schema.blocks}
    current_block: Optional[Block] = None

    lines = src.splitlines()
    for idx, original_line in enumerate(lines):
        line_no = idx + 1
        line_wo_comment = _strip_comment(original_line)
        raw = line_wo_comment.rstrip("\n")

        if _is_blank(raw):
            continue

        if _is_block_header(raw):
            name = _parse_block_name(raw)
            if name is None:
                diags.append(
                    Diagnostic(
                        code=Err.E0001_UNEXPECTED,
                        severity=Severity.ERROR,
                        message="Invalid block header. Expected @BLOCK_NAME using uppercase letters/digits/underscores.",
                        span=_make_span_for_line(line_no, 1, max(1, len(original_line))),
                        suggestions=["Example: @WORLD", "Example: @MECHANICS"],
                    )
                )
                current_block = None
                continue

            if enforce_known_blocks and (not schema.allow_extra_blocks) and (name not in block_specs):
                diags.append(
                    Diagnostic(
                        code=Err.E0102_UNKNOWN_BLOCK,
                        severity=Severity.ERROR,
                        message=f"Unknown block @{name}.",
                        span=_make_span_for_line(line_no, 1, max(1, len(original_line))),
                        context={"known_blocks": [b.name for b in schema.blocks]},
                        suggestions=["Use one of the known blocks, e.g. @INTENT, @WORLD, @MECHANICS, @RULES, @OUTPUT."],
                    )
                )
                current_block = None
                continue

            if program.has_block(name):
                prev = program.block(name)
                diags.append(
                    Diagnostic(
                        code=Err.E0100_DUP_BLOCK,
                        severity=Severity.ERROR,
                        message=f"Duplicate block @{name}.",
                        span=_make_span_for_line(line_no, 1, max(1, len(original_line))),
                        context={
                            "previous_header_span": None if prev is None or prev.header_span is None else prev.header_span.to_dict()
                        },
                        suggestions=[f"Remove the duplicate @{name} block, or merge its keys into the first one."],
                    )
                )
                # keep parsing inside the new block anyway (but it won't be stored)
                current_block = Block(name=name, header_span=_make_span_for_line(line_no, 1, max(1, len(original_line))))
                continue

            current_block = Block(name=name, header_span=_make_span_for_line(line_no, 1, max(1, len(original_line))))
            program.blocks[name] = current_block
            continue

        # Non-header lines must be inside a block
        if current_block is None:
            diags.append(
                Diagnostic(
                    code=Err.E0001_UNEXPECTED,
                    severity=Severity.ERROR,
                    message="Key-value line found outside of any block. Start a block with @BLOCK_NAME.",
                    span=_make_span_for_line(line_no, 1, max(1, len(original_line))),
                    suggestions=["Add a block header above this line, e.g. @INTENT"],
                )
            )
            continue

        kv = _parse_key_value_line(raw)
        if kv is None:
            diags.append(
                Diagnostic(
                    code=Err.E0001_UNEXPECTED,
                    severity=Severity.ERROR,
                    message="Invalid line in block: expected 'key: value'.",
                    span=_make_span_for_line(line_no, 1, max(1, len(original_line))),
                    context={"block": current_block.name},
                    suggestions=["Example: title: \"My Game\"", "Example: actions: [move, attack, wait]"],
                )
            )
            continue

        key, raw_value, colon_index = kv

        # Unknown key enforcement (optional)
        if enforce_known_keys and current_block.name in block_specs:
            bs = block_specs[current_block.name]
            fs_map = field_spec_map(bs)  # type: ignore[arg-type]
            if (not bs.allow_extra_keys) and (key not in fs_map):
                diags.append(
                    Diagnostic(
                        code=Err.E0001_UNEXPECTED,
                        severity=Severity.ERROR,
                        message=f"Unknown key '{key}' in @{current_block.name}.",
                        span=_make_span_for_line(line_no, 1, max(1, len(original_line))),
                        context={"block": current_block.name, "known_keys": list(fs_map.keys())},
                        suggestions=["Fix the key name or move it into the correct block."],
                    )
                )
                continue

        if key in current_block.kvs:
            prev = current_block.kvs[key]
            diags.append(
                Diagnostic(
                    code=Err.E0101_DUP_KEY,
                    severity=Severity.ERROR,
                    message=f"Duplicate key '{key}' in @{current_block.name}.",
                    span=_make_span_for_line(line_no, 1, max(1, len(original_line))),
                    context={"block": current_block.name, "previous_key_span": None if prev.span is None else prev.span.to_dict()},
                    suggestions=["Remove the duplicate key, or rename it if you meant a different field."],
                )
            )
            continue

        # Parse value
        # Determine value column start (1-based): after colon plus at least one char
        # We trimmed raw_value, so approximate value span at colon_index + 2
        value_start_col = colon_index + 2
        v, v_diags = _parse_value(raw_value, line_no, value_start_col)
        diags.extend(v_diags)
        if v is None:
            continue

        kv_span = _make_span_for_line(line_no, 1, max(1, len(original_line)))
        current_block.kvs[key] = KeyValue(key=key, value=v, span=kv_span)

    ok = len([d for d in diags if d.severity == Severity.ERROR]) == 0
    return CompilationResult(ok=ok, errors=diags, ast=program if ok else None, version=schema.version)


def parse_file(
    path: str,
    *,
    schema: Schema = GPTLANG_V1_SCHEMA,
    enforce_known_blocks: bool = True,
    enforce_known_keys: bool = False,
    encoding: str = "utf-8",
) -> CompilationResult:
    with open(path, "r", encoding=encoding) as f:
        return parse_string(
            f.read(),
            schema=schema,
            enforce_known_blocks=enforce_known_blocks,
            enforce_known_keys=enforce_known_keys,
        )
