"""Faithfulness judges used by the offline evaluation.

The runtime citation checker decides what the user sees; the eval judge should be
*independent* of it, otherwise the system grades its own homework. Order of
preference, picked automatically unless --judge is given:

  ragas   RAGAS Faithfulness (+ context recall, factual correctness) with an LLM judge
  llm     our own versioned `faithfulness_judge` prompt, claim by claim
  nli     NLI cross-encoder entailment
  lexical deterministic term-support (keyless CI; weakest, but stable)
"""

from __future__ import annotations

import math
import os
from typing import Protocol

from ..generation.citations import LexicalSupportChecker, NLISupportChecker, split_sentences
from ..ingest.chunker import Chunk
from ..prompts import get_prompt


class Judge(Protocol):
    name: str

    def faithfulness(self, question: str, answer: str, contexts: list[Chunk]) -> float: ...


def _claims(answer: str) -> list[str]:
    import re

    return [re.sub(r"\s*\[\d+\]", "", s).strip() for s in split_sentences(answer) if s.strip()]


class CheckerJudge:
    """Claim-level faithfulness with any SupportChecker: a claim counts as supported if
    ANY retrieved chunk supports it (RAGAS semantics: support by the context as a whole)."""

    def __init__(self, checker, name: str):
        self.checker = checker
        self.name = name

    def faithfulness(self, question: str, answer: str, contexts: list[Chunk]) -> float:
        claims = _claims(answer)
        if not claims:
            return float("nan")
        ok = sum(any(self.checker.check(c, ch).supported for ch in contexts) for c in claims)
        return ok / len(claims)


class LLMJudge:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = get_prompt("faithfulness_judge")
        self.name = f"llm-judge:{llm.name}:{self.prompt.ref}"

    def faithfulness(self, question: str, answer: str, contexts: list[Chunk]) -> float:
        claims = _claims(answer)
        if not claims:
            return float("nan")
        ctx = "\n\n".join(c.text for c in contexts)
        ok = 0
        for claim in claims:
            verdict = self.llm.complete(self.prompt["system"], self.prompt.render("user", context=ctx, claim=claim))
            ok += verdict.strip().upper().startswith("SUPPORTED")
        return ok / len(claims)


class RagasJudge:
    """Batch judge: RAGAS computes metrics over the whole run in one call."""

    def __init__(self, provider: str = "anthropic", model: str | None = None):
        from ragas.llms import LangchainLLMWrapper

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            lc = ChatAnthropic(model=model or "claude-sonnet-5-5", temperature=0, max_tokens=2048)
        else:
            from langchain_openai import ChatOpenAI

            lc = ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
        self.llm = LangchainLLMWrapper(lc)
        self.name = f"ragas:{provider}:{model or 'default'}"

    def evaluate(self, rows: list[dict]) -> list[dict]:
        """rows: {question, answer, contexts: [str], reference}. Returns per-row scores."""
        import warnings

        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ds = EvaluationDataset.from_list([
            {"user_input": r["question"], "response": r["answer"],
             "retrieved_contexts": r["contexts"] or [""], "reference": r["reference"]} for r in rows])
        result = evaluate(ds, metrics=[Faithfulness(), LLMContextRecall(), FactualCorrectness()],
                          llm=self.llm, show_progress=False, raise_exceptions=False)
        df = result.to_pandas()
        # column names vary slightly across RAGAS versions, e.g. "factual_correctness(mode=f1)"
        cols = {name: next((c for c in df.columns if c.startswith(name)), None)
                for name in ("faithfulness", "context_recall", "factual_correctness")}
        return [{name: (_f(rec[col]) if col else float("nan")) for name, col in cols.items()}
                for _, rec in df.iterrows()]


def _f(x) -> float:
    try:
        x = float(x)
        return x if not math.isinf(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def build_judge(kind: str, settings, idf: dict[str, float]):
    if kind == "auto":
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            try:
                import ragas  # noqa: F401

                kind = "ragas"
            except ImportError:
                kind = "llm"
        else:
            try:
                import sentence_transformers  # noqa: F401

                kind = "nli" if settings.get("profile") != "offline" else "lexical"
            except ImportError:
                kind = "lexical"
    if kind == "ragas":
        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
        return RagasJudge(provider)
    if kind == "llm":
        from ..generation.llm import build_llm

        cfg = dict(settings.generation)
        if cfg["backend"] == "extractive":
            cfg["backend"] = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
            cfg["model"] = "claude-sonnet-5-5" if cfg["backend"] == "anthropic" else "gpt-4o-mini"
        return LLMJudge(build_llm(cfg, "INSUFFICIENT_EVIDENCE"))
    if kind == "nli":
        g = settings.guardrails
        return CheckerJudge(NLISupportChecker(g["nli_model"], g.get("nli_entailment_threshold", 0.5)), "nli")
    if kind == "lexical":
        # Deliberately stricter than the runtime gate (0.55), so offline eval is not
        # just echoing the runtime checker's own decisions.
        return CheckerJudge(LexicalSupportChecker(0.7, idf), "lexical@0.7")
    raise ValueError(f"Unknown judge {kind}")
