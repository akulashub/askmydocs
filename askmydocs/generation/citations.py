"""Citation enforcement.

After generation, every answer sentence is checked:
1. it carries at least one [n] citation,
2. every [n] refers to a passage that was actually in the prompt,
3. at least one cited passage *supports* the sentence (NLI entailment, or lexical
   support offline). The best-supporting paragraph becomes the precise citation.
Unsupported sentences are dropped; if too little of the answer survives, the system
refuses instead of returning a plausible-sounding but ungrounded answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ..ingest.chunker import Chunk
from ..retrieval.text import tokenize

CITE = re.compile(r"\[(\d+)\]")
# A sentence ends at . ! or ? optionally followed by its citation markers, e.g.
# "... a response. [1] Next" or "... a response [1]. Next".
_BOUNDARY = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)([.!?](?:\s*\[\d+\])*)\s+(?=[A-Z`*(\-])")


@dataclass
class Support:
    supported: bool
    score: float
    paragraph_index: int  # index into chunk.paragraphs of the best-supporting paragraph


@dataclass
class CheckedSentence:
    text: str                          # sentence without citation markers
    cited: list[int]                   # passage numbers as written by the model
    valid_cited: list[int]             # cited numbers that exist in the context
    supported: bool
    support_score: float
    evidence: list[dict] = field(default_factory=list)  # [{n, chunk_id, citation, paragraph}]


class SupportChecker(Protocol):
    name: str

    def check(self, claim: str, chunk: Chunk) -> Support: ...


class LexicalSupportChecker:
    """Fraction of the claim's IDF-weighted content terms found in a single paragraph
    (with its neighbours, since claims often span a sentence boundary)."""

    name = "lexical-support-v1"

    def __init__(self, threshold: float = 0.55, idf: dict[str, float] | None = None):
        self.threshold = threshold
        self.idf = idf or {}

    def check(self, claim: str, chunk: Chunk) -> Support:
        terms = set(tokenize(claim))
        if not terms:
            return Support(False, 0.0, 0)
        total = sum(self.idf.get(t, 1.0) for t in terms)
        paras = chunk.paragraphs or [chunk.text]
        ranked = []
        for i in range(len(paras)):
            window = set(tokenize(" ".join(paras[max(0, i - 1): i + 2])))
            own = set(tokenize(paras[i]))
            score = sum(self.idf.get(t, 1.0) for t in terms & window) / total
            own_score = sum(self.idf.get(t, 1.0) for t in terms & own) / total
            ranked.append((score, own_score, -i))  # tie-break: the paragraph holding most of it
        score, _, neg_i = max(ranked)
        return Support(score >= self.threshold, round(score, 4), -neg_i)


class NLISupportChecker:
    """Entailment probability from an NLI cross-encoder, premise = one paragraph
    (+ neighbours), hypothesis = the claim. Max over paragraphs."""

    def __init__(self, model: str, threshold: float = 0.5):
        from sentence_transformers import CrossEncoder  # pip install .[models]

        self.model = CrossEncoder(model)
        self.threshold = threshold
        self.name = f"nli:{model}"
        config = getattr(self.model, "config", None) or self.model.model.config
        labels = {v.lower(): int(k) for k, v in config.id2label.items()}
        self.entail_idx = labels.get("entailment", 1)

    def check(self, claim: str, chunk: Chunk) -> Support:
        import numpy as np

        paras = chunk.paragraphs or [chunk.text]
        premises = [" ".join(paras[max(0, i - 1): i + 2]) for i in range(len(paras))]
        logits = np.asarray(self.model.predict([(p, claim) for p in premises]))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        ent = probs[:, self.entail_idx]
        i = int(ent.argmax())
        return Support(bool(ent[i] >= self.threshold), round(float(ent[i]), 4), i)


def split_sentences(answer: str) -> list[str]:
    marked = _BOUNDARY.sub(lambda m: m.group(1) + "\x00", answer.strip())
    return [p.strip() for p in marked.split("\x00") if p.strip()]


def check_answer(answer: str, context: list[Chunk], checker: SupportChecker) -> list[CheckedSentence]:
    """`context[i]` is passage number i+1 in the prompt."""
    results: list[CheckedSentence] = []
    for sent in split_sentences(answer):
        cited = [int(n) for n in CITE.findall(sent)]
        text = CITE.sub("", sent).strip()
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        valid = [n for n in dict.fromkeys(cited) if 1 <= n <= len(context)]
        evidence, best_score, supported = [], 0.0, False
        for n in valid:
            chunk = context[n - 1]
            sup = checker.check(text, chunk)
            best_score = max(best_score, sup.score)
            if sup.supported:
                supported = True
                span = chunk.spans[sup.paragraph_index] if chunk.spans else None
                evidence.append({
                    "n": n, "chunk_id": chunk.chunk_id, "citation": chunk.citation(span),
                    "paragraph": (chunk.paragraphs or [chunk.text])[sup.paragraph_index],
                    "score": sup.score,
                })
        results.append(CheckedSentence(text, cited, valid, supported, best_score, evidence))
    return results


def build_support_checker(cfg: dict, idf: dict[str, float] | None = None) -> SupportChecker:
    if cfg.get("support_backend") == "nli":
        return NLISupportChecker(cfg["nli_model"], cfg.get("nli_entailment_threshold", 0.5))
    return LexicalSupportChecker(cfg.get("lexical_support_threshold", 0.55), idf)
