"""Example systems-under-test.

In a real project these would call your LLM/agent. Here they're deterministic
rule-based classifiers so the demo runs offline — and so we can simulate a
*regression* (v2 is deliberately worse) to show the gate catching it.
"""
from __future__ import annotations

KEYWORDS = {
    "billing": ["charged", "refund", "invoice", "payment", "price", "subscription"],
    "bug": ["error", "crash", "500", "broken", "doesn't work", "fails"],
    "feature_request": ["add", "feature", "would be nice", "dark mode", "support for"],
    "account": ["log in", "login", "password", "reset", "account", "sign in"],
}


def triage_v1(text: str) -> str:
    """Decent classifier: scores every category and picks the best."""
    text_l = text.lower()
    best, best_score = "other", 0
    for label, kws in KEYWORDS.items():
        score = sum(1 for k in kws if k in text_l)
        if score > best_score:
            best, best_score = label, score
    return best


def triage_v2_regressed(text: str) -> str:
    """A 'prompt change' that broke billing detection (dropped key terms)."""
    text_l = text.lower()
    broken = dict(KEYWORDS)
    broken["billing"] = ["payment"]  # oops — removed 'charged'/'refund'/'invoice'
    best, best_score = "other", 0
    for label, kws in broken.items():
        score = sum(1 for k in kws if k in text_l)
        if score > best_score:
            best, best_score = label, score
    return best
