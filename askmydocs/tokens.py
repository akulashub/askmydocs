"""Token counting.

Modes (set from `chunking.tokenizer` in settings):
  auto     tiktoken cl100k_base when its encoding can be loaded, else regex
  tiktoken require tiktoken
  regex    deterministic regex approximation (within ~10% of BPE for English prose)

The offline profile pins `regex` so chunk boundaries, and therefore eval metrics,
are identical on every machine regardless of whether tiktoken can download its
encoding file.
"""

from __future__ import annotations

import re
from functools import lru_cache

_APPROX = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
_MODE = "auto"


def set_tokenizer(mode: str) -> None:
    global _MODE
    if mode not in {"auto", "tiktoken", "regex"}:
        raise ValueError(f"Unknown tokenizer mode {mode!r}")
    _MODE = mode


@lru_cache(maxsize=1)
def _tiktoken():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline, or tiktoken not installed
        return None


def _encoder():
    if _MODE == "regex":
        return None
    enc = _tiktoken()
    if enc is None and _MODE == "tiktoken":
        raise RuntimeError("chunking.tokenizer=tiktoken but the cl100k_base encoding is unavailable")
    return enc


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    # Long alphabetic words split into ~2 BPE tokens; others count as one.
    return sum(1 + (len(t) > 8) for t in _APPROX.findall(text))


def tokenizer_name() -> str:
    return "tiktoken:cl100k_base" if _encoder() is not None else "regex-approx"
