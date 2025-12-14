# gptlang_prompts.py

from __future__ import annotations

import json
from typing import Any, Dict, List


def diagnostics_prompt(diagnostics: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    GPT mode: explain deterministic compiler errors + propose concrete fixes.
    Input: list of Diagnostic dicts (from CompilationResult.to_dict()).
    Output: human-readable explanation + exact patch suggestions.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a compiler diagnostics engine for GPTLANG v1.\n"
                "You are given deterministic compiler diagnostics (errors) from the frontend.\n"
                "Your job:\n"
                "1) Explain what each error means in plain language.\n"
                "2) Provide concrete corrections (exact lines/blocks to add/change).\n"
                "Constraints:\n"
                "- Do NOT invent new language features.\n"
                "- Do NOT generate Python code.\n"
                "- Keep fixes minimal and directly tied to the error list.\n"
            ),
        },
        {
            "role": "user",
            "content": "Diagnostics JSON:\n" + json.dumps(diagnostics, indent=2),
        },
    ]


def codegen_prompt(program_ast: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    GPT mode: codegen backend.
    Input: validated Program AST as JSON.
    Output: a single Python file content (no markdown).
    """
    return [
        {
            "role": "system",
            "content": (
                "You are the GPTLANG v1 compiler backend.\n"
                "Input: a validated GPTLANG v1 AST (JSON).\n"
                "Output: EXACTLY ONE Python file as plain text.\n"
                "No markdown. No triple backticks. No explanations.\n\n"
                "Hard requirements for generated Python:\n"
                "- Must be runnable with: python gen_game.py\n"
                "- Must define: def main(): and call it under if __name__ == '__main__':\n"
                "- Must be offline: no network, no file I/O beyond standard printing.\n"
                "- Must be deterministic: use WORLD.seed if randomness is used.\n"
                "- Render must be ASCII (text grid / text UI). Do NOT use pygame.\n"
                "- Implement a small game loop consistent with MECHANICS.loop:\n"
                "  - turn_based: prompt for action each turn; stop at max_turns or win/lose.\n"
                "  - real_time: simulate ticks using time.sleep(tick_ms/1000) and non-blocking input is optional.\n"
                "- Implement actions from RULES.actions. Unsupported actions must not appear.\n"
                "- Enforce lose_condition and win_condition.\n"
                "- Keep the implementation compact (hackathon demo), but correct.\n\n"
                "Interpretation guidance:\n"
                "- Use INTENT.title as the game title.\n"
                "- WORLD.setting + WORLD.tone influences flavor text only.\n"
                "- MECHANICS.genre influences simple mechanics (e.g. score, obstacles) but keep minimal.\n"
                "- difficulty affects numbers (damage, score target, enemy spawn chance) deterministically.\n"
                "- win_condition:\n"
                "  - reach_goal: player reaches a goal tile on a small grid.\n"
                "  - survive_timer: survive N ticks/turns.\n"
                "  - score_threshold: reach a score target.\n"
                "- lose_condition:\n"
                "  - hp_zero: HP <= 0 ends game.\n"
                "  - timeout: if turns/ticks exceed max allowed.\n\n"
                "Safety:\n"
                "- No eval/exec.\n"
                "- No imports beyond: random, time, sys, math (optional).\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Validated GPTLANG AST (JSON):\n"
                + json.dumps(program_ast, indent=2)
                + "\n\n"
                "Generate the Python file now."
            ),
        },
    ]
