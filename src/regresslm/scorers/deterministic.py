"""Deterministic scorers — fast, free, and the backbone of any eval suite.

Reach for an LLM judge only for genuinely subjective dimensions; everything
checkable (labels, formats, numbers, required content) should be deterministic.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..schema import Score
from .base import ScorerContext


def _as_text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, sort_keys=True, default=str)


class ExactMatch:
    name = "exact_match"

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        ref = ctx.case.reference
        ok = _as_text(output).strip() == _as_text(ref).strip()
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail="" if ok else f"expected {ref!r}, got {output!r}")


class Contains:
    """Pass if output contains every required substring (expect['substrings'])."""
    name = "contains"

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        needles = ctx.case.expect.get("substrings", [])
        if ctx.case.expect.get("contains"):
            needles = [ctx.case.expect["contains"]]
        text = _as_text(output).lower()
        hits = [n for n in needles if n.lower() in text]
        value = len(hits) / len(needles) if needles else 1.0
        ok = value == 1.0
        missing = [n for n in needles if n not in hits]
        return Score(scorer=self.name, value=value, passed=ok,
                     detail="" if ok else f"missing: {missing}")


class RegexMatch:
    name = "regex"

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        pattern = ctx.case.expect.get("regex")
        if not pattern:
            return Score(scorer=self.name, value=1.0, passed=True, detail="no regex configured")
        ok = re.search(pattern, _as_text(output)) is not None
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail="" if ok else f"no match for /{pattern}/")


class NumericTolerance:
    """Compare a numeric output to reference within expect['tol'] (absolute)."""
    name = "numeric_tolerance"

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        tol = float(ctx.case.expect.get("tol", 0.0))
        try:
            got = float(output)
            exp = float(ctx.case.reference)
        except (TypeError, ValueError):
            return Score(scorer=self.name, value=0.0, passed=False, detail="non-numeric")
        ok = abs(got - exp) <= tol
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail="" if ok else f"|{got}-{exp}|>{tol}")


class JsonValid:
    """Pass if output is valid JSON and (optionally) matches expect['schema']."""
    name = "json_valid"

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        try:
            obj = output if isinstance(output, (dict, list)) else json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return Score(scorer=self.name, value=0.0, passed=False, detail="invalid JSON")
        schema = ctx.case.expect.get("schema")
        if schema:
            try:
                import jsonschema
                jsonschema.validate(obj, schema)
            except Exception as e:  # jsonschema.ValidationError or missing dep
                return Score(scorer=self.name, value=0.0, passed=False, detail=f"schema: {e}")
        return Score(scorer=self.name, value=1.0, passed=True)


class LabelMatch:
    """Classification accuracy: output label == expect['label']."""
    name = "label_match"

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        expected = ctx.case.expect.get("label", ctx.case.reference)
        got = _as_text(output).strip().lower()
        ok = got == _as_text(expected).strip().lower()
        return Score(scorer=self.name, value=1.0 if ok else 0.0, passed=ok,
                     detail="" if ok else f"expected '{expected}', got '{output}'")
