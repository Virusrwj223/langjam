## Purpose

GPTLANG v1 is a **GPT-first programming language** for describing small games in a **formal, sectioned DSL** stored in `.llm` files. This makes GPT act as a **compiler backend**, while correctness/structure is enforced by a deterministic frontend.

---

## File & Folder Convention

The compiler expects a **project folder** containing **exactly one** `.llm` file:

```
my_game/
  main.llm
```

On success, the compiler writes the generated Python file **into the same folder**:

```
my_game/
  main.llm
  gen_game.py
```

---

## GPTLANG v1 Syntax

A `.llm` file is made of **blocks**.

- A block starts with `@BLOCK_NAME`
- Inside a block, you write `key: value` lines
- Blocks end at the next `@...` header or end-of-file
- Comments start with `#` (outside quotes)

### Minimal Example

```text
@INTENT
title: "67 Vibes Dungeon"
patch: 1
mode: brainrot_arena
audience_vibe: chaotic
```

### Values

Supported value types:

- **String**: `"..."` (single-line)
- **Int**: `123`, `-5`
- **Bool**: `true` / `false`
- **Ident**: `move`, `brainrot_arena`, `turn_based`
- **List**: `[move, attack, wait]`
- **Map**: `{ key: "value", other: 123 }` (supported, but optional for v1 usage)

---

## Required Blocks (v1)

These blocks must exist **exactly once**:

- `@INTENT`
- `@WORLD`
- `@MECHANICS`
- `@RULES`
- `@OUTPUT`

Optional block:

- `@CONSTRAINTS`

---

## Block Specifications (v1)

### `@INTENT` (required)

| Key             | Type   | Constraints                                       |
| --------------- | ------ | ------------------------------------------------- |
| `title`         | string | non-empty                                         |
| `patch`         | int    | ≥ 1                                               |
| `mode`          | ident  | `vibe_quest \| grindset_runner \| brainrot_arena` |
| `audience_vibe` | ident  | `cozy \| sweaty \| chaotic \| cinematic`          |

---

### `@WORLD` (required)

| Key       | Type   | Constraints                                          |
| --------- | ------ | ---------------------------------------------------- |
| `setting` | string | non-empty                                            |
| `tone`    | ident  | `wholesome \| unhinged \| spooky \| chill \| absurd` |
| `seed`    | int    | 0..999999                                            |

---

### `@MECHANICS` (required)

| Key              | Type  | Constraints                                                |
| ---------------- | ----- | ---------------------------------------------------------- |
| `genre`          | ident | `roguelike \| platformer \| narrative \| puzzle \| arcade` |
| `loop`           | ident | `turn_based \| real_time`                                  |
| `player_hp`      | int   | 1..999                                                     |
| `win_condition`  | ident | `reach_goal \| survive_timer \| score_threshold`           |
| `lose_condition` | ident | `hp_zero \| timeout`                                       |

---

### `@RULES` (required)

| Key          | Type        | Constraints                                                   |
| ------------ | ----------- | ------------------------------------------------------------- |
| `actions`    | list[ident] | non-empty; each in `move, jump, dash, attack, interact, wait` |
| `difficulty` | ident       | `ez \| normal \| cracked`                                     |
| `tick_ms`    | int         | 16..1000 (required if loop = `real_time`)                     |
| `max_turns`  | int         | 1..10000 (required if loop = `turn_based`)                    |

---

### `@OUTPUT` (required)

| Key          | Type  | Constraints                              |
| ------------ | ----- | ---------------------------------------- |
| `target`     | ident | must be `python`                         |
| `entrypoint` | ident | must be `main`                           |
| `render`     | ident | `ascii` (v1); `pygame` is rejected in v1 |
| `logging`    | bool  | `true \| false`                          |

---

### `@CONSTRAINTS` (optional)

| Key             | Type | Constraints                                  |
| --------------- | ---- | -------------------------------------------- |
| `no_network`    | bool | if present, must be `true` for demo pipeline |
| `max_loc`       | int  | 200..2000                                    |
| `max_runtime_s` | int  | 1..30                                        |

---

## Deterministic Validation Rules

The compiler enforces:

1. **Structure**
   - Only one of each block (duplicates are errors)
   - `key: value` format in blocks
   - Duplicate keys in a block are errors
2. **Schema**
   - Required blocks must exist
   - Required keys must exist
   - Types must match exactly
   - Enums must match allowed values
   - Int values must be within range
3. **Cross-field constraints**
   - If `MECHANICS.loop = real_time` then `RULES.tick_ms` is required
   - If `MECHANICS.loop = turn_based` then `RULES.max_turns` is required
4. **Codegen gate**
   - If any errors exist, **no Python is generated**

Diagnostics are emitted in structured JSON with error codes (e.g., `E1000`, `E1100`).

---

## OpenRouter Configuration

For the purpose of the hackathon, a free-tier API key is provided for usage. The compiler reads configuration from environment variables (typically via a `.env` file):

```env
LLM_PROVIDER=openrouter
LLM_MODEL=openai/gpt-oss-20b:free

OPENROUTER_API_KEY=YOUR_KEY_HERE
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
```

Notes:

- `LLM_PROVIDER` must be exactly `openrouter`
- `OPENROUTER_BASE_URL` must point to the chat completions endpoint shown above

## How to Run

### 1) Create a project folder

```
my_game/
```

### 2) Add exactly one `.llm` file

Save as `my_game/main.llm` (any name is fine as long as it ends in `.llm` and is the only one in the folder).

Example: **mental sum game**

```text
@INTENT
title: "Brainrot Mental Sums: 67 Mode"
patch: 1
mode: grindset_runner
audience_vibe: sweaty

@WORLD
setting: "A neon study room where every correct sum gives you +vibes"
tone: absurd
seed: 67

@MECHANICS
genre: arcade
loop: turn_based
player_hp: 5
win_condition: score_threshold
lose_condition: hp_zero

@RULES
actions: [interact, wait]
max_turns: 25
difficulty: normal

@OUTPUT
target: python
entrypoint: main
render: ascii
logging: true

@CONSTRAINTS
no_network: true
max_loc: 700
max_runtime_s: 15
```

### 3) Compile

From your repo root (where `cllam.py` lives):

```bash
python cllam.py my_game/
```

If valid, you will get:

```
my_game/gen_game.py
```

### 4) Run the generated game

```bash
python my_game/gen_game.py
```

---

## Troubleshooting

### `Unsupported LLM_PROVIDER: ''`

Your environment variables are not loaded. Ensure `.env` is exported into the shell before running.

### `OpenRouter returned a non-JSON response`

Most commonly:

- `OPENROUTER_BASE_URL` is wrong (must end with `/chat/completions`)
- A proxy/captive portal returned HTML
- Network interception

### Parse/Validation errors

The compiler prints deterministic JSON diagnostics to stderr. Fix the `.llm` file and re-run.
