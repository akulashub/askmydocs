"""Citation enforcement: the system must refuse rather than return ungrounded text."""

from askmydocs.generation.citations import LexicalSupportChecker, check_answer, split_sentences


class FakeLLM:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def complete(self, system, user):
        return self.text


def test_split_sentences_keeps_citations_attached():
    s = split_sentences("First fact. [1] Second fact [2]. Third, e.g. `x.y()` here. [1][3] Tail")
    assert s == ["First fact. [1]", "Second fact [2].", "Third, e.g. `x.y()` here. [1][3]", "Tail"]


def test_supported_answer_passes_with_paragraph_citation(rag):
    ans = rag.ask("Where must Field be imported from?")
    assert not ans.refused
    assert ans.sources and all(s.citation.startswith("tutorial/") for s in ans.sources)
    assert any("pydantic" in s.paragraph for s in ans.sources)
    assert ans.trace["prompt"].startswith("answer_with_citations@")


def test_off_topic_question_is_refused_before_generation(rag):
    ans = rag.ask("What is the capital of Australia?")
    assert ans.refused and ans.refusal_reason == "no_relevant_context"
    assert ans.raw_answer == ""  # the LLM was never called


def test_hallucinated_answer_is_refused(rag, monkeypatch):
    monkeypatch.setattr(rag, "llm", FakeLLM(
        "FastAPI was written in Rust for maximum speed [1]. It requires a PostgreSQL database [2]."))
    ans = rag.ask("How do I add CORS middleware?")
    assert ans.refused and ans.refusal_reason == "citations_not_supported"
    assert ans.supported_fraction == 0.0


def test_uncited_and_out_of_range_citations_are_rejected(rag, monkeypatch):
    monkeypatch.setattr(rag, "llm", FakeLLM("CORS is configured with CORSMiddleware. [9]"))
    ans = rag.ask("How do I add CORS middleware?")
    assert ans.refused
    assert ans.sentences[0]["valid_cited"] == []


def test_partially_supported_answer_drops_bad_sentence_or_refuses(rag, monkeypatch):
    ctx = rag.retrieve_only("What is the default value of max_age in CORSMiddleware?")
    n = next(i for i, c in enumerate(ctx, 1) if "Defaults to `600`" in c.chunk.text)
    good = f"max_age sets a maximum time in seconds for browsers to cache CORS responses and defaults to 600. [{n}]"
    bad = f"FastAPI caches every response in Redis for one hour. [{n}]"
    monkeypatch.setattr(rag, "llm", FakeLLM(f"{good} {bad}"))
    ans = rag.ask("What is the default value of max_age in CORSMiddleware?")
    # 1 of 2 sentences supported (50%) is below min_supported_fraction (80%): refuse.
    assert ans.refused and ans.refusal_reason == "citations_not_supported"


def test_model_refusal_token_is_respected(rag, monkeypatch):
    monkeypatch.setattr(rag, "llm", FakeLLM("INSUFFICIENT_EVIDENCE"))
    ans = rag.ask("How do I add CORS middleware?")
    assert ans.refused and ans.refusal_reason == "model_declined"


def test_lexical_checker_rejects_unrelated_claim(rag):
    chunk = next(c for c in rag.chunks if c.doc_id == "tutorial/cors.md")
    checker = LexicalSupportChecker(0.55, rag.idf)
    assert not checker.check("Django migrations are created with makemigrations.", chunk).supported
    res = check_answer("An origin is the combination of protocol, domain, and port. [1]", [chunk], checker)
    assert res[0].supported and res[0].evidence[0]["citation"].startswith("tutorial/cors.md#")
