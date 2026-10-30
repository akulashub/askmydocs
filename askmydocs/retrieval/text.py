"""Shared text normalisation for lexical retrieval and support checking.

Technical docs are full of identifiers (`add_task`, `BackgroundTasks`,
`response_model_exclude_unset`). We index both the identifier and its parts so a
query for "background tasks" matches `BackgroundTasks` and an exact query for
`add_task` still gets an exact-term boost."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from as is are was were be been being
it its this that these those i you he she we they them your our their my me do does did doing have has
had having can could should would will shall may might must not no yes so such than too very just also
how what which who whom when where why there here into over under about between through during before
after above below up down out off again further once all any both each few more most other some own same
only s t don use using used get gets set sets e g etc way like one two make makes made need needs
""".split())


def _stem(word: str) -> str:
    # Minimal, deterministic suffix stripping; good enough for English docs.
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            if suf in ("ies", "ied"):
                return word[: -len(suf)] + "y"
            return word[: -len(suf)]
    return word


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        low = tok.lower()
        parts = [p.lower() for piece in tok.split("_") for p in _CAMEL.findall(piece)]
        compound = len(parts) > 1
        if compound:
            out.append(low)  # exact identifier
        for p in parts or [low]:
            if keep_stopwords or (p not in STOPWORDS and len(p) > 1):
                out.append(_stem(p))
    return out


def content_terms(text: str) -> set[str]:
    return set(tokenize(text))
