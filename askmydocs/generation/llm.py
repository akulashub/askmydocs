"""LLM backends behind one `complete(system, user)` interface.

`ExtractiveLLM` is a deterministic, offline stand-in that obeys the same output
contract as the real prompt (inline [n] citations, refusal token). It answers by
selecting the context sentences that best cover the question, so the whole pipeline,
including citation enforcement, runs in tests and keyless CI.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from ..retrieval.text import tokenize

_CTX = re.compile(r"^\[(\d+)\] \(source: .*?\)\n(.*?)(?=^\[\d+\] \(source:|\Z)", re.S | re.M)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*(])")
_FENCED = re.compile(r"```.*?```", re.S)


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700):
        import anthropic  # pip install .[anthropic]

        self.client = anthropic.Anthropic()
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, system=system, max_tokens=self.max_tokens, temperature=self.temperature,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


class OpenAILLM:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 700):
        from openai import OpenAI  # pip install .[openai]

        self.client = OpenAI()
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"openai:{model}"

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (resp.choices[0].message.content or "").strip()


class ExtractiveLLM:
    name = "extractive-v1"

    def __init__(self, refusal_token: str = "INSUFFICIENT_EVIDENCE", max_sentences: int = 3,
                 min_coverage: float = 0.34, idf: dict[str, float] | None = None, focus_terms: int = 1):
        self.refusal_token = refusal_token
        self.focus_terms = focus_terms
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage
        self.idf = idf or {}

    def complete(self, system: str, user: str) -> str:
        q_match = re.search(r"Question:\s*(.*?)\n\s*\nContext passages:", user, re.S)
        question = q_match.group(1).strip() if q_match else ""
        q_terms = set(tokenize(question))
        if not q_terms:
            return self.refusal_token
        total = sum(self.idf.get(t, 1.0) for t in q_terms)
        scored: list[tuple[float, int, int, str]] = []
        passages: dict[int, set[str]] = {}
        for m in _CTX.finditer(user):
            n, body = int(m.group(1)), _FENCED.sub("", m.group(2))
            passages[n] = set(tokenize(m.group(0)))
            for para in body.split("\n\n"):
                if para.lstrip().startswith(("```", "|", "    ")):
                    continue  # don't quote code/tables as prose
                for j, sent in enumerate(_SENT.split(" ".join(para.split()))):
                    terms = set(tokenize(sent))
                    if len(terms) < 3:
                        continue
                    cov = sum(self.idf.get(t, 1.0) for t in q_terms & terms) / total
                    # position prior: earlier passages were ranked higher
                    scored.append((cov - 0.01 * n, n, j, sent.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] < self.min_coverage:
            return self.refusal_token
        picked: list[tuple[float, int, int, str]] = []
        seen: set[str] = set()
        for s in scored:
            if len(picked) == self.max_sentences or s[0] < scored[0][0] * 0.6:
                break
            key = s[3].lower()
            if key not in seen:
                picked.append(s)
                seen.add(key)
        # Focus check (what an LLM does implicitly): the question's most specific terms
        # must occur in the passages we are about to cite. "What port does Redis use?"
        # must not be answered from a passage about Traefik's port.
        focus = sorted(q_terms, key=lambda t: (-self.idf.get(t, 1.0), t))[: self.focus_terms]
        cited_terms = set().union(*(passages.get(s[1], set()) for s in picked))
        if any(t not in cited_terms for t in focus):
            return self.refusal_token
        picked.sort(key=lambda x: (x[1], x[2]))
        out = []
        for _, n, _, sent in picked:
            sent = sent.rstrip(":")
            if not sent.endswith((".", "!", "?")):
                sent += "."
            out.append(f"{sent} [{n}]")
        return " ".join(out)


def build_llm(cfg: dict, refusal_token: str, idf: dict[str, float] | None = None) -> LLM:
    backend = cfg["backend"]
    if backend == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (or use --profile offline).")
        return AnthropicLLM(cfg["model"], cfg.get("temperature", 0.0), cfg.get("max_tokens", 700))
    if backend == "openai":
        return OpenAILLM(cfg["model"], cfg.get("temperature", 0.0), cfg.get("max_tokens", 700))
    if backend == "extractive":
        return ExtractiveLLM(refusal_token=refusal_token, idf=idf)
    raise ValueError(f"Unknown generation backend: {backend}")
