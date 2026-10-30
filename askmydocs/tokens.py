"""Token counting. Uses tiktoken (cl100k_base) when its encoding is available,
otherwise a deterministic regex approximation (within ~10% for English prose)."""

from __future__ import annotations

import re
from functools import lru_cache

_APPROX = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline, or tiktoken not installed
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # Long alphabetic words split into ~2 BPE tokens; others count as one.
    return sum(1 + (len(t) > 8) for t in _APPROX.findall(text))


def tokenizer_name() -> str:
    return "tiktoken:cl100k_base" if _encoder() is not None else "regex-approx"
