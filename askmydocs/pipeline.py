"""The question-answering pipeline, orchestrated as a LangGraph state machine:

    retrieve -> rerank -> gate --(relevant)--> generate -> verify --(grounded)--> END
                            \\--(irrelevant)--> refuse                \\--(not grounded)--> refuse

Every answer records the prompt version, model names, per-stage scores and
latencies, so any answer (or eval regression) can be traced to its configuration.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from .config import Settings, load_settings
from .embeddings import build_embedder
from .generation.citations import CheckedSentence, build_support_checker, check_answer
from .generation.llm import build_llm
from .indexer import compute_idf, ensure_index
from .ingest.chunker import Chunk
from .prompts import get_prompt
from .retrieval.rerankers import build_reranker
from .retrieval.retrievers import BM25Retriever, Candidate, HybridRetriever, VectorRetriever
from .store import ChromaVectorStore, read_chunks, read_manifest

DOMAIN = "the FastAPI web framework (official tutorial and advanced user guide)"


@dataclass
class Source:
    n: int
    citation: str
    chunk_id: str
    paragraph: str


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    sources: list[Source] = field(default_factory=list)
    sentences: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)   # reranked context, in prompt order
    raw_answer: str = ""
    supported_fraction: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def context_chunk_ids(self) -> list[str]:
        return [r["chunk_id"] for r in self.retrieved]

    def to_dict(self) -> dict:
        return asdict(self)


class State(TypedDict, total=False):
    question: str
    candidates: list[Candidate]
    context: list[Candidate]
    raw_answer: str
    checked: list[CheckedSentence]
    answer: Answer
    timings: dict[str, float]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, log=print):
        self.settings = settings or load_settings()
        s = self.settings
        paths = ensure_index(s, log)
        self.manifest = read_manifest(paths)
        self.chunks: list[Chunk] = read_chunks(paths)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.idf = compute_idf(self.chunks)

        self.prompt = get_prompt(s.generation["prompt_id"])
        self.ctx_prompt = get_prompt("context_block")
        self.refusal_token = self.prompt["refusal_token"]

        embedder = build_embedder(s.embeddings)
        store = ChromaVectorStore(paths, s.index["collection"])
        self.retriever = HybridRetriever(VectorRetriever(store, embedder, self.by_id),
                                         BM25Retriever(self.chunks), s.retrieval)
        self.reranker = build_reranker(s.reranker, self.idf)
        self.llm = build_llm(s.generation, self.refusal_token, self.idf)
        self.checker = build_support_checker(s.guardrails, self.idf)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ nodes
    def _retrieve(self, st: State) -> State:
        t = time.perf_counter()
        cands = self.retriever.search(st["question"])
        return {"candidates": cands, "timings": {**st.get("timings", {}), "retrieve": time.perf_counter() - t}}

    def _rerank(self, st: State) -> State:
        t = time.perf_counter()
        ctx = self.reranker.rerank(st["question"], list(st["candidates"]), self.settings.retrieval["rerank_top_n"])
        return {"context": ctx, "timings": {**st["timings"], "rerank": time.perf_counter() - t}}

    def _route_after_rerank(self, st: State) -> str:
        ctx = st["context"]
        if not ctx:
            return "refuse"
        top = ctx[0].scores.get("rerank", ctx[0].score)
        return "generate" if top >= self.settings.guardrails["min_rerank_score"] else "refuse"

    def _render_context(self, ctx: list[Candidate]) -> str:
        return "\n".join(self.ctx_prompt.render(
            "template", n=i, source=c.chunk.doc_id, section=c.chunk.section or c.chunk.title, text=c.chunk.text)
            for i, c in enumerate(ctx, start=1))

    def _generate(self, st: State) -> State:
        t = time.perf_counter()
        system = self.prompt.render("system", domain=DOMAIN, refusal_token=self.refusal_token)
        user = self.prompt.render("user", question=st["question"], context=self._render_context(st["context"]),
                                  refusal_token=self.refusal_token)
        raw = self.llm.complete(system, user)
        return {"raw_answer": raw, "timings": {**st["timings"], "generate": time.perf_counter() - t}}

    def _verify(self, st: State) -> State:
        t = time.perf_counter()
        raw = st["raw_answer"]
        if self.refusal_token in raw:
            checked: list[CheckedSentence] = []
        else:
            checked = check_answer(raw, [c.chunk for c in st["context"]], self.checker)
        return {"checked": checked, "timings": {**st["timings"], "verify": time.perf_counter() - t}}

    def _route_after_verify(self, st: State) -> str:
        checked = st["checked"]
        if not checked:
            return "refuse"
        frac = sum(s.supported for s in checked) / len(checked)
        return "finalize" if frac >= self.settings.guardrails["min_supported_fraction"] else "refuse"

    def _finalize(self, st: State) -> State:
        checked = st["checked"]
        drop = self.settings.guardrails.get("drop_unsupported_sentences", True)
        kept = [s for s in checked if s.supported or not drop]
        sources: dict[str, Source] = {}
        parts: list[str] = []
        for s in kept:
            nums = []
            for ev in s.evidence:
                key = ev["citation"]
                if key not in sources:
                    sources[key] = Source(len(sources) + 1, key, ev["chunk_id"], ev["paragraph"])
                if sources[key].n not in nums:
                    nums.append(sources[key].n)
            parts.append(s.text + "".join(f" [{n}]" for n in nums))
        ans = self._make_answer(st, refused=False, text=" ".join(parts), sources=list(sources.values()))
        return {"answer": ans}

    def _refuse(self, st: State) -> State:
        if "raw_answer" not in st:
            reason = "no_relevant_context"
        elif self.refusal_token in st["raw_answer"]:
            reason = "model_declined"
        else:
            reason = "citations_not_supported"
        text = ("I can't answer that from the indexed documentation: the retrieved passages "
                "don't contain enough supporting evidence.")
        return {"answer": self._make_answer(st, refused=True, text=text, reason=reason)}

    def _make_answer(self, st: State, refused: bool, text: str, sources: list[Source] | None = None,
                     reason: str = "") -> Answer:
        checked = st.get("checked", [])
        frac = (sum(s.supported for s in checked) / len(checked)) if checked else 0.0
        ctx = st.get("context", [])
        return Answer(
            question=st["question"], answer=text, refused=refused, refusal_reason=reason,
            sources=sources or [],
            sentences=[asdict(s) for s in checked],
            retrieved=[{"n": i, "chunk_id": c.chunk.chunk_id, "citation": c.chunk.citation(),
                        "scores": {k: round(v, 4) for k, v in c.scores.items()}, "ranks": c.ranks}
                       for i, c in enumerate(ctx, start=1)],
            raw_answer=st.get("raw_answer", ""),
            supported_fraction=round(frac, 4),
            trace={
                "prompt": self.prompt.ref,
                "embedder": self.manifest.get("embedder"),
                "reranker": self.reranker.name,
                "llm": self.llm.name,
                "support_checker": self.checker.name,
                "retrieval_mode": self.settings.retrieval["mode"],
                "profile": self.settings.get("profile"),
                "timings_ms": {k: round(v * 1000, 1) for k, v in st.get("timings", {}).items()},
                "candidates": [c.chunk.chunk_id for c in st.get("candidates", [])],
            },
        )

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:  # pragma: no cover - langgraph is a core dependency
            return None
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("rerank", self._rerank)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)
        g.add_node("finalize", self._finalize)
        g.add_node("refuse", self._refuse)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "rerank")
        g.add_conditional_edges("rerank", self._route_after_rerank, {"generate": "generate", "refuse": "refuse"})
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", self._route_after_verify, {"finalize": "finalize", "refuse": "refuse"})
        g.add_edge("finalize", END)
        g.add_edge("refuse", END)
        return g.compile()

    def ask(self, question: str) -> Answer:
        state: State = {"question": question.strip(), "timings": {}}
        if self.graph is not None:
            return self.graph.invoke(state)["answer"]
        # Fallback without LangGraph: identical control flow.
        state.update(self._retrieve(state))
        state.update(self._rerank(state))
        if self._route_after_rerank(state) == "refuse":
            return self._refuse(state)["answer"]
        state.update(self._generate(state))
        state.update(self._verify(state))
        nxt = self._route_after_verify(state)
        return (self._finalize(state) if nxt == "finalize" else self._refuse(state))["answer"]

    def retrieve_only(self, question: str) -> list[Candidate]:
        st: State = {"question": question, "timings": {}}
        st.update(self._retrieve(st))
        st.update(self._rerank(st))
        return st["context"]
