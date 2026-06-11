"""LLM-as-judge scoring.

Use for subjective dimensions (helpfulness, faithfulness, tone) that
deterministic checks can't capture. Two implementations:

  • MockJudge — deterministic, offline. Used in tests/CI so the suite never
    depends on a live model or API key.
  • LLMJudge  — calls the Claude API with a rubric and parses a 1–5 score.

The judge returns a normalized 0..1 value plus a short rationale.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from ..schema import Score
from .base import ScorerContext

DEFAULT_RUBRIC = (
    "You are a strict evaluator. Given a task INPUT, an optional REFERENCE answer, "
    "and a model OUTPUT, rate the OUTPUT from 1 (terrible) to 5 (excellent) on "
    "correctness and helpfulness. Respond ONLY as JSON: "
    '{"score": <1-5>, "reason": "<one sentence>"}.'
)


def _normalize_1_5(raw: int | float) -> float:
    return max(0.0, min(1.0, (float(raw) - 1.0) / 4.0))


class MockJudge:
    """Deterministic stand-in. Scores by lexical overlap with the reference so
    tests are stable and meaningful without a real model."""
    name = "llm_judge"

    def __init__(self, pass_threshold: float = 0.5):
        self.pass_threshold = pass_threshold

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        ref = ctx.case.reference
        out_text = output if isinstance(output, str) else json.dumps(output, default=str)
        if not ref:
            # No reference -> reward non-empty, structured-looking answers.
            value = 1.0 if out_text.strip() else 0.0
        else:
            ref_tokens = set(re.findall(r"\w+", str(ref).lower()))
            out_tokens = set(re.findall(r"\w+", out_text.lower()))
            if not ref_tokens:
                value = 1.0 if out_tokens else 0.0
            else:
                value = len(ref_tokens & out_tokens) / len(ref_tokens)
        value = round(value, 4)
        return Score(scorer=self.name, value=value, passed=value >= self.pass_threshold,
                     detail=f"mock overlap={value}")


class LLMJudge:
    """Real judge backed by the Claude API.

    `complete` is an injectable function (prompt -> text) so the judge is
    testable and provider-agnostic. The default wires up the Anthropic SDK.
    """
    name = "llm_judge"

    def __init__(
        self,
        rubric: str = DEFAULT_RUBRIC,
        pass_threshold: float = 0.5,
        model: str = "claude-haiku-4-5",
        complete: Callable[[str], str] | None = None,
    ):
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.model = model
        self._complete = complete or self._default_complete

    def _default_complete(self, prompt: str) -> str:
        # Lazily import so the package works without the anthropic SDK installed.
        from anthropic import Anthropic
        client = Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def _build_prompt(self, output: Any, ctx: ScorerContext) -> str:
        return (
            f"{self.rubric}\n\n"
            f"INPUT:\n{ctx.case.input}\n\n"
            f"REFERENCE:\n{ctx.case.reference}\n\n"
            f"OUTPUT:\n{output}\n"
        )

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        text = self._complete(self._build_prompt(output, ctx))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return Score(scorer=self.name, value=0.0, passed=False, detail="judge returned no JSON")
        try:
            data = json.loads(match.group(0))
            value = _normalize_1_5(data.get("score", 1))
            reason = str(data.get("reason", ""))[:200]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return Score(scorer=self.name, value=0.0, passed=False, detail=f"parse error: {e}")
        return Score(scorer=self.name, value=value, passed=value >= self.pass_threshold, detail=reason)
