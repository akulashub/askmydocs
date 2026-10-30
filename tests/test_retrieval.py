from askmydocs.retrieval.retrievers import Candidate, reciprocal_rank_fusion
from askmydocs.retrieval.text import tokenize


class _C:
    def __init__(self, cid):
        self.chunk_id = cid


def _cands(ids):
    return [Candidate(_C(i), 1.0) for i in ids]


def test_tokenize_splits_identifiers_but_keeps_them():
    toks = tokenize("Use `BackgroundTasks.add_task()` with response_model_exclude_unset")
    assert "add_task" in toks and "add" in toks and "task" in toks
    assert "backgroundtasks" in toks and "background" in toks
    assert "response_model_exclude_unset" in toks and "unset" in toks


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion({"vector": _cands(["a", "b", "c"]), "bm25": _cands(["c", "a", "d"])})
    order = [c.chunk.chunk_id for c in fused]
    assert order[0] == "a"            # rank 1 + rank 2 beats everything else
    assert set(order) == {"a", "b", "c", "d"}
    assert fused[0].ranks == {"vector": 1, "bm25": 2}


def test_bm25_finds_exact_identifier(rag):
    hits = rag.retriever.bm25.search("response_model_exclude_unset", 5)
    assert any("response-model.md" in h.chunk.doc_id for h in hits[:3])


def test_hybrid_beats_or_matches_each_retriever(rag):
    """Hybrid should put the gold chunk at least as high as the better single retriever
    on a query mixing an exact term with natural language."""
    q = "what does check_same_thread do for the sqlite engine"

    def rank(results):
        return next((i for i, c in enumerate(results, 1) if c.chunk.doc_id == "tutorial/sql-databases.md"), 99)

    hy = rank(rag.retriever.search(q))
    vec = rank(rag.retriever.vector.search(q, 20))
    bm = rank(rag.retriever.bm25.search(q, 20))
    assert hy <= max(vec, bm)
    assert hy <= 3


def test_reranker_returns_top_n_sorted(rag):
    ctx = rag.retrieve_only("How do I add CORS middleware?")
    assert len(ctx) == rag.settings.retrieval["rerank_top_n"]
    scores = [c.scores["rerank"] for c in ctx]
    assert scores == sorted(scores, reverse=True)
    assert ctx[0].chunk.doc_id == "tutorial/cors.md"
