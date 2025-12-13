# gptlang_parse.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from gptlang_ast import (
    BlockStmt,
    CanonKey,
    CanonSection,
    FieldStmt,
    File,
    NameRef,
    Position,
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
    make_block,
    make_field,
)

# ============================================================
# 0) Slang alias maps (raw -> canonical)
# ============================================================

SECTION_ALIASES: Dict[str, str] = {
    # Canonical -> user-facing slang are documented elsewhere;
    # here we map raw -> canonical.
    "VIBE_CHECK": CanonSection.INTENT.value,
    "WORLD_BUILD": CanonSection.WORLD.value,
    "HOW_IT_HITS": CanonSection.MECHANICS.value,
    "NO_CAP_RULES": CanonSection.RULES.value,
    "SHIP_IT": CanonSection.OUTPUT.value,
    "DONT_BE_SUS": CanonSection.CONSTRAINTS.value,
    "RECEIPTS": CanonSection.METADATA.value,
    "PLAYBOOK": CanonSection.PROGRAM.value,
}

# Keys can optionally be interpreted *in context* of a section.
# This keeps slang short without ambiguity.
KEY_ALIASES_BY_SECTION: Dict[str, Dict[str, str]] = {
    CanonSection.INTENT.value: {
        "title": CanonKey.NAME.value,
        "patch": CanonKey.VERSION.value,
        "mode": CanonKey.STAGE.value,
    },
    CanonSection.WORLD.value: {
        "setting": CanonKey.SETTING.value,
        "loot": CanonKey.ASSETS.value,
    },
    CanonSection.MECHANICS.value: {
        "turns": CanonKey.TURNS.value,
        "moves": CanonKey.ACTIONS.value,
        "dub_condition": CanonKey.WIN_CONDITION.value,
    },
    CanonSection.OUTPUT.value: {
        "session_dsl": CanonKey.DSL_NAME.value,
        "print_templates": CanonKey.EMIT_TEMPLATES.value,
    },
    CanonSection.PROGRAM.value: {
        "session_dsl": CanonKey.DSL.value,
        "gameplan": CanonKey.STRATEGY.value,
    },
}

# Some keys can also appear as canonical raw keys (power users).
CANONICAL_KEY_FALLBACK: Dict[str, str] = {k.value: k.value for k in CanonKey if k != CanonKey.UNKNOWN}


def canon_section(raw: str) -> str:
    return SECTION_ALIASES.get(raw, CanonSection.UNKNOWN.value)


def canon_key(section_canon: str, raw: str) -> str:
    # first try section-specific slang
    m = KEY_ALIASES_BY_SECTION.get(section_canon, {})
    if raw in m:
        return m[raw]
    # then allow canonical keys directly
    if raw in CANONICAL_KEY_FALLBACK:
        return raw
    return CanonKey.UNKNOWN.value


# ============================================================
# 1) Diagnostics
# ============================================================

@dataclass(frozen=True)
class Suggestion:
    title: str
    patch: str


@dataclass(frozen=True)
class Diagnostic:
    code: str                 # E0001, etc.
    severity: str             # "error" | "warning"
    message: str
    span: Span
    suggestions: List[Suggestion]


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    file_ast: Optional[File]
    diagnostics: List[Diagnostic]


def _mk_diag(code: str, message: str, span: Span, suggestions: Optional[List[Suggestion]] = None) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity="error",
        message=message,
        span=span,
        suggestions=suggestions or [],
    )


# ============================================================
# 2) Tokenizer (Lexer) with spans
# ============================================================

class TokKind(str, Enum):
    AT = "@"
    LBRACE = "{"
    RBRACE = "}"
    LBRACK = "["
    RBRACK = "]"
    COLON = ":"
    SEMI = ";"
    COMMA = ","

    IDENT = "IDENT"
    STRING = "STRING"
    INT = "INT"
    BOOL = "BOOL"

    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    kind: TokKind
    lexeme: str
    span: Span
    literal: Optional[Union[str, int, bool]] = None


class Lexer:
    def __init__(self, text: str, source_name: Optional[str] = None):
        self.text = text
        self.n = len(text)
        self.i = 0
        self.line = 1
        self.col = 1
        self.source_name = source_name

    # ---------- position helpers ----------

    def _pos(self) -> Position:
        return Position(line=self.line, col=self.col, index=self.i)

    def _advance(self, k: int = 1) -> None:
        for _ in range(k):
            if self.i >= self.n:
                return
            ch = self.text[self.i]
            self.i += 1
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1

    def _peek(self, off: int = 0) -> str:
        j = self.i + off
        if j >= self.n:
            return "\0"
        return self.text[j]

    def _match(self, s: str) -> bool:
        if self.text.startswith(s, self.i):
            self._advance(len(s))
            return True
        return False

    # ---------- skipping whitespace & comments ----------

    def _skip_ws_and_comments(self) -> Optional[Diagnostic]:
        while True:
            ch = self._peek()
            # whitespace
            if ch in (" ", "\t", "\r", "\n"):
                self._advance(1)
                continue

            # line comment: # ... \n
            if ch == "#":
                while self._peek() not in ("\n", "\0"):
                    self._advance(1)
                continue

            # block comment: /* ... */
            if self._peek() == "/" and self._peek(1) == "*":
                start = self._pos()
                self._advance(2)
                while True:
                    if self._peek() == "\0":
                        end = self._pos()
                        return _mk_diag("E0004", "Unclosed block comment '/* ... */'", Span(start, end))
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._advance(2)
                        break
                    self._advance(1)
                continue

            return None

    # ---------- token readers ----------

    def next_token(self) -> Tuple[Token, Optional[Diagnostic]]:
        diag = self._skip_ws_and_comments()
        if diag is not None:
            # Return a synthetic EOF so parser can terminate cleanly.
            eof_pos = self._pos()
            return Token(TokKind.EOF, "", Span(eof_pos, eof_pos)), diag

        start = self._pos()
        ch = self._peek()

        if ch == "\0":
            return Token(TokKind.EOF, "", Span(start, start)), None

        # single-char punctuators
        single = {
            "@": TokKind.AT,
            "{": TokKind.LBRACE,
            "}": TokKind.RBRACE,
            "[": TokKind.LBRACK,
            "]": TokKind.RBRACK,
            ":": TokKind.COLON,
            ";": TokKind.SEMI,
            ",": TokKind.COMMA,
        }
        if ch in single:
            self._advance(1)
            end = self._pos()
            return Token(single[ch], ch, Span(start, end)), None

        # string
        if ch == '"':
            return self._read_string()

        # int (optional leading -)
        if ch == "-" or ch.isdigit():
            if ch == "-" and not self._peek(1).isdigit():
                # treat '-' as unexpected token in v1
                self._advance(1)
                end = self._pos()
                return Token(TokKind.IDENT, "-", Span(start, end), literal="-"), _mk_diag(
                    "E0001", "Unexpected token '-'", Span(start, end)
                )
            return self._read_int()

        # ident / bool
        if ch.isalpha() or ch == "_":
            return self._read_ident_or_bool()

        # unknown
        self._advance(1)
        end = self._pos()
        return Token(TokKind.IDENT, ch, Span(start, end), literal=ch), _mk_diag(
            "E0001", f"Unexpected character {ch!r}", Span(start, end)
        )

    def _read_ident_or_bool(self) -> Tuple[Token, Optional[Diagnostic]]:
        start = self._pos()
        j = self.i
        while True:
            c = self._peek(j - self.i)
            if c.isalnum() or c == "_":
                j += 1
                continue
            break
        lex = self.text[self.i:j]
        self._advance(len(lex))
        end = self._pos()

        if lex == "true":
            return Token(TokKind.BOOL, lex, Span(start, end), literal=True), None
        if lex == "false":
            return Token(TokKind.BOOL, lex, Span(start, end), literal=False), None
        return Token(TokKind.IDENT, lex, Span(start, end), literal=lex), None

    def _read_int(self) -> Tuple[Token, Optional[Diagnostic]]:
        start = self._pos()
        j = self.i
        if self._peek() == "-":
            j += 1
        while self.text[j:j+1].isdigit():
            j += 1
        lex = self.text[self.i:j]
        self._advance(len(lex))
        end = self._pos()
        try:
            n = int(lex)
        except ValueError:
            return Token(TokKind.INT, lex, Span(start, end)), _mk_diag("E0001", f"Bad integer literal {lex!r}", Span(start, end))
        return Token(TokKind.INT, lex, Span(start, end), literal=n), None

    def _read_string(self) -> Tuple[Token, Optional[Diagnostic]]:
        start = self._pos()
        assert self._peek() == '"'
        self._advance(1)  # consume opening "

        out_chars: List[str] = []
        while True:
            ch = self._peek()
            if ch == "\0":
                end = self._pos()
                return Token(TokKind.STRING, "", Span(start, end), literal=""), _mk_diag(
                    "E0002", "Unterminated string literal", Span(start, end)
                )
            if ch == "\n":
                # disallow multiline strings in v1
                end = self._pos()
                return Token(TokKind.STRING, "", Span(start, end), literal=""), _mk_diag(
                    "E0002", "String literal cannot contain newline (v1)", Span(start, end)
                )
            if ch == '"':
                self._advance(1)
                break
            if ch == "\\":
                self._advance(1)
                esc = self._peek()
                if esc == "\0":
                    end = self._pos()
                    return Token(TokKind.STRING, "", Span(start, end), literal=""), _mk_diag(
                        "E0002", "Unterminated escape sequence in string", Span(start, end)
                    )
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                if esc in mapping:
                    out_chars.append(mapping[esc])
                    self._advance(1)
                else:
                    # consume unknown escape as literal char (but error)
                    out_chars.append(esc)
                    esc_start = self._pos()
                    self._advance(1)
                    esc_end = self._pos()
                    # Return error but still produce token (parser continues)
                    # We'll attach the error to this token.
                    end = self._pos()
                    tok = Token(TokKind.STRING, "", Span(start, end), literal="".join(out_chars))
                    return tok, _mk_diag("E0001", f"Unknown escape '\\{esc}'", Span(esc_start, esc_end))
            else:
                out_chars.append(ch)
                self._advance(1)

        end = self._pos()
        s = "".join(out_chars)
        lexeme = self.text[start.index:end.index]  # includes quotes
        return Token(TokKind.STRING, lexeme, Span(start, end), literal=s), None


# ============================================================
# 3) Parser (recursive descent) with spans + recovery
# ============================================================

class Parser:
    def __init__(self, text: str, source_name: Optional[str] = None):
        self.lexer = Lexer(text, source_name=source_name)
        self.source_name = source_name
        self.diagnostics: List[Diagnostic] = []

        self.cur: Token
        self._advance()  # prime first token

    def _advance(self) -> None:
        tok, diag = self.lexer.next_token()
        self.cur = tok
        if diag is not None:
            self.diagnostics.append(diag)

    def _at(self, kind: TokKind) -> bool:
        return self.cur.kind == kind

    def _expect(self, kind: TokKind, code: str, msg: str) -> Token:
        if self.cur.kind == kind:
            tok = self.cur
            self._advance()
            return tok
        # error span: current token span
        self.diagnostics.append(_mk_diag(code, msg, self.cur.span))
        # recovery: do not consume if EOF, else consume one token to progress
        tok = self.cur
        if self.cur.kind != TokKind.EOF:
            self._advance()
        return tok

    def parse(self) -> ParseResult:
        sections: List[Section] = []
        seen_sections: set[str] = set()

        while not self._at(TokKind.EOF):
            if not self._at(TokKind.AT):
                # skip until next '@' or EOF
                self.diagnostics.append(_mk_diag(
                    "E0001",
                    f"Expected '@' to start a section, found {self.cur.kind.value}",
                    self.cur.span
                ))
                self._sync_to_section_start()
                continue

            sec = self._parse_section()
            if sec is None:
                self._sync_to_section_start()
                continue

            # Duplicate section raw name check (v1: error)
            raw = sec.name.raw
            if raw in seen_sections:
                self.diagnostics.append(_mk_diag(
                    "E0005",
                    f"Duplicate section @{raw}",
                    sec.span
                ))
            else:
                seen_sections.add(raw)
                sections.append(sec)

        ok = len([d for d in self.diagnostics if d.severity == "error"]) == 0
        file_ast = File(sections=sections, source_name=self.source_name) if ok else None
        return ParseResult(ok=ok, file_ast=file_ast, diagnostics=self.diagnostics)

    def _sync_to_section_start(self) -> None:
        while not self._at(TokKind.EOF) and not self._at(TokKind.AT):
            self._advance()

    def _sync_to_stmt_end(self) -> None:
        # skip until ; or } or @ or EOF
        while not self._at(TokKind.EOF) and not self._at(TokKind.SEMI) and not self._at(TokKind.RBRACE) and not self._at(TokKind.AT):
            self._advance()
        if self._at(TokKind.SEMI):
            self._advance()

    # ---------- section ----------

    def _parse_section(self) -> Optional[Section]:
        at_tok = self._expect(TokKind.AT, "E0001", "Expected '@'")

        if self.cur.kind != TokKind.IDENT:
            self.diagnostics.append(_mk_diag("E0001", "Expected section name after '@'", self.cur.span))
            return None

        name_tok = self.cur
        self._advance()

        # canonicalize section name
        sec_raw = name_tok.lexeme
        sec_canon = canon_section(sec_raw)
        name_ref = NameRef(raw=sec_raw, canon=sec_canon)

        lbrace = self._expect(TokKind.LBRACE, "E0001", f"Expected '{{' after section name @{sec_raw}")

        body: List[Statement] = []
        # parse statements until }
        while not self._at(TokKind.EOF) and not self._at(TokKind.RBRACE):
            st = self._parse_statement(section_canon=sec_canon)
            if st is None:
                self._sync_to_stmt_end()
            else:
                body.append(st)

        rbrace = self._expect(TokKind.RBRACE, "E0004", f"Unclosed section @{sec_raw}: expected '}}'")

        span = Span(at_tok.span.start, rbrace.span.end if rbrace else self.cur.span.end)
        return Section(name=name_ref, body=body, span=span)

    # ---------- statement ----------

    def _parse_statement(self, section_canon: str) -> Optional[Statement]:
        if self.cur.kind != TokKind.IDENT:
            self.diagnostics.append(_mk_diag("E0001", "Expected statement key (identifier)", self.cur.span))
            return None

        key_tok = self.cur
        self._advance()

        key_raw = key_tok.lexeme
        key_canon = canon_key(section_canon, key_raw)
        key_ref = NameRef(raw=key_raw, canon=key_canon)

        # Decide field vs block
        if self._at(TokKind.COLON):
            colon = self.cur
            self._advance()
            val = self._parse_value()
            if val is None:
                self.diagnostics.append(_mk_diag("E0001", "Expected value after ':'", self.cur.span))
                return None

            semi = self._expect(TokKind.SEMI, "E0003", "Missing ';' after field statement")
            stmt_span = Span(key_tok.span.start, semi.span.end)
            return make_field(key_ref, val, stmt_span)

        if self._at(TokKind.LBRACE):
            lbrace = self.cur
            self._advance()
            fields: List[FieldStmt] = []
            # blocks only allow field statements in v1
            while not self._at(TokKind.EOF) and not self._at(TokKind.RBRACE):
                f = self._parse_block_field(section_canon=section_canon, block_key_canon=key_canon)
                if f is None:
                    self._sync_to_stmt_end()
                else:
                    fields.append(f)
            rbrace = self._expect(TokKind.RBRACE, "E0004", "Unclosed block: expected '}'")
            stmt_span = Span(key_tok.span.start, rbrace.span.end)
            return make_block(key_ref, fields, stmt_span)

        # If neither ":" nor "{", it's a syntax error.
        self.diagnostics.append(_mk_diag(
            "E0001",
            "Expected ':' for field or '{' for block after key",
            self.cur.span
        ))
        return None

    def _parse_block_field(self, section_canon: str, block_key_canon: str) -> Optional[FieldStmt]:
        # In v1, a block contains only field statements: key: value;
        if self.cur.kind != TokKind.IDENT:
            self.diagnostics.append(_mk_diag("E0001", "Expected field key inside block", self.cur.span))
            return None
        key_tok = self.cur
        self._advance()

        # NOTE: For blocks, you may want a different alias map later (block-specific).
        # For now: treat as canonical/unknown under same section.
        raw = key_tok.lexeme
        # In blocks, keys are schema-defined: keep canon == raw unless you later add block-specific aliases.
        key_ref = NameRef(raw=raw, canon=raw)


        self._expect(TokKind.COLON, "E0001", "Expected ':' after field key inside block")
        val = self._parse_value()
        if val is None:
            self.diagnostics.append(_mk_diag("E0001", "Expected value after ':'", self.cur.span))
            return None
        semi = self._expect(TokKind.SEMI, "E0003", "Missing ';' after field statement")
        span = Span(key_tok.span.start, semi.span.end)
        return FieldStmt(key=key_ref, value=val, span=span)

    # ---------- value ----------

    def _parse_value(self) -> Optional[Value]:
        if self._at(TokKind.STRING):
            tok = self.cur
            self._advance()
            assert isinstance(tok.literal, str)
            return Value(ValueKind.STRING, VString(tok.literal), tok.span)

        if self._at(TokKind.INT):
            tok = self.cur
            self._advance()
            assert isinstance(tok.literal, int)
            return Value(ValueKind.INT, VInt(tok.literal), tok.span)

        if self._at(TokKind.BOOL):
            tok = self.cur
            self._advance()
            assert isinstance(tok.literal, bool)
            return Value(ValueKind.BOOL, VBool(tok.literal), tok.span)

        if self._at(TokKind.IDENT):
            tok = self.cur
            self._advance()
            return Value(ValueKind.IDENT, VIdent(tok.lexeme), tok.span)

        if self._at(TokKind.LBRACK):
            return self._parse_list()

        if self._at(TokKind.LBRACE):
            return self._parse_object()

        self.diagnostics.append(_mk_diag("E0001", "Expected a value", self.cur.span))
        return None

    def _parse_list(self) -> Optional[Value]:
        lbr = self._expect(TokKind.LBRACK, "E0001", "Expected '['")
        items: List[Value] = []

        # empty list allowed
        if self._at(TokKind.RBRACK):
            rbr = self.cur
            self._advance()
            return Value(ValueKind.LIST, VList(items), Span(lbr.span.start, rbr.span.end))

        while not self._at(TokKind.EOF):
            v = self._parse_value()
            if v is None:
                # try to recover to comma or ]
                while not self._at(TokKind.EOF) and not self._at(TokKind.COMMA) and not self._at(TokKind.RBRACK):
                    self._advance()
            else:
                items.append(v)

            if self._at(TokKind.COMMA):
                self._advance()
                continue
            if self._at(TokKind.RBRACK):
                rbr = self.cur
                self._advance()
                return Value(ValueKind.LIST, VList(items), Span(lbr.span.start, rbr.span.end))

            # neither comma nor ], error
            self.diagnostics.append(_mk_diag("E0001", "Expected ',' or ']' in list", self.cur.span))
            # attempt recovery
            while not self._at(TokKind.EOF) and not self._at(TokKind.RBRACK):
                self._advance()

        # EOF before closing ]
        self.diagnostics.append(_mk_diag("E0004", "Unclosed list: expected ']'", Span(lbr.span.start, self.cur.span.end)))
        return None

    def _parse_object(self) -> Optional[Value]:
        lbr = self._expect(TokKind.LBRACE, "E0001", "Expected '{'")
        props: Dict[str, Value] = {}

        if self._at(TokKind.RBRACE):
            rbr = self.cur
            self._advance()
            return Value(ValueKind.OBJECT, VObject(props), Span(lbr.span.start, rbr.span.end))

        while not self._at(TokKind.EOF):
            if self.cur.kind != TokKind.IDENT:
                self.diagnostics.append(_mk_diag("E0001", "Expected key (identifier) in object", self.cur.span))
                # recover to comma or }
                while not self._at(TokKind.EOF) and not self._at(TokKind.COMMA) and not self._at(TokKind.RBRACE):
                    self._advance()
            else:
                key_tok = self.cur
                self._advance()
                self._expect(TokKind.COLON, "E0001", "Expected ':' after object key")
                v = self._parse_value()
                if v is not None:
                    props[key_tok.lexeme] = v

            if self._at(TokKind.COMMA):
                self._advance()
                continue
            if self._at(TokKind.RBRACE):
                rbr = self.cur
                self._advance()
                return Value(ValueKind.OBJECT, VObject(props), Span(lbr.span.start, rbr.span.end))

            self.diagnostics.append(_mk_diag("E0001", "Expected ',' or '}' in object", self.cur.span))
            while not self._at(TokKind.EOF) and not self._at(TokKind.RBRACE):
                self._advance()

        self.diagnostics.append(_mk_diag("E0004", "Unclosed object: expected '}'", Span(lbr.span.start, self.cur.span.end)))
        return None


# ============================================================
# 4) Public API
# ============================================================

def parse_gptlang(text: str, source_name: Optional[str] = None) -> ParseResult:
    """
    Deterministic parse (syntax + structure only).
    If parse errors exist: ok=False and file_ast=None (v1).
    """
    p = Parser(text, source_name=source_name)
    return p.parse()


# ============================================================
# 5) Tiny smoke test (optional)
# ============================================================

# if __name__ == "__main__":
#     sample = r'''
#     # main.llm (slangy)
#     @VIBE_CHECK {
#       title: "Coin Flip Arena";
#       patch: 1;
#       mode: meta;
#     }

#     @WORLD_BUILD {
#       setting: "neon exchange";
#       loot: [BTC, ETH];
#     }

#     @HOW_IT_HITS {
#       turns: 50;
#       moves: [trade, hold];
#       dub_condition: "profit max";
#     }

#     @SHIP_IT {
#       session_dsl: VibeExchangeSession;
#       print_templates: true;
#     }
#     '''
#     res = parse_gptlang(sample, source_name="main.llm")
#     print("ok:", res.ok)
#     for d in res.diagnostics:
#         print(d.code, d.message, f"@ {d.span.start.line}:{d.span.start.col}")
#     if res.file_ast:
#         for s in res.file_ast.sections:
#             print("SECTION", s.name.raw, "->", s.name.canon)
