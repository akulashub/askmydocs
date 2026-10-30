from pathlib import Path

from askmydocs.config import ROOT
from askmydocs.ingest.chunker import chunk_document
from askmydocs.ingest.loaders import Block, Document, load_corpus, load_markdown
from askmydocs.tokens import count_tokens

CORPUS = ROOT / "data" / "corpus" / "fastapi"


def _doc(n_paragraphs: int, words: int = 60) -> Document:
    sentence = " ".join(["the dependency system resolves every parameter"] * 2) + "."
    blocks = [Block(f"Paragraph {i}. " + " ".join([sentence] * (words // 12)), i * 2 + 1, i * 2 + 1,
                    "Doc > Sec", "sec") for i in range(n_paragraphs)]
    return Document("doc.md", "Doc", "markdown", blocks)


def test_markdown_loader_strips_includes_and_anchors(tmp_path: Path):
    md = tmp_path / "x.md"
    md.write_text("# Title { #title }\n\nIntro text here.\n\n{* ../src/a.py hl[1] *}\n\n"
                  "## Sub `Thing` { #sub-thing }\n\n/// tip\n\nUse it wisely.\n\n///\n\n```python\nx = 1\n\ny = 2\n```\n")
    doc = load_markdown(md, "x.md")
    texts = [b.text for b in doc.blocks]
    assert doc.title == "Title"
    assert "Intro text here." in texts
    assert not any("{*" in t or "{ #" in t for t in texts)
    tip = next(b for b in doc.blocks if "wisely" in b.text)
    assert tip.text.startswith("Tip: ") and tip.anchor == "sub-thing" and tip.section == "Title > Sub Thing"
    code = next(b for b in doc.blocks if b.kind == "code")
    assert "x = 1" in code.text and "y = 2" in code.text  # blank line inside fence kept in one block


def test_chunks_respect_token_budget_and_overlap():
    chunks = chunk_document(_doc(40), target_tokens=600, max_tokens=800, min_tokens=120, overlap_tokens=100)
    assert len(chunks) > 3
    assert all(c.token_count <= 800 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        # b starts with text carried over from the end of a, at most ~100 tokens of it
        head = b.paragraphs[0]
        assert head.split(". ")[-1] in a.text, "consecutive chunks must overlap"
        carried = [p for p in b.paragraphs if p in a.text or a.text.endswith(p)]
        assert 0 < sum(count_tokens(p) for p in carried) <= 100 + 5


def test_paragraph_spans_are_aligned():
    for c in chunk_document(_doc(25)):
        assert len(c.spans) == len(c.paragraphs) == len(c.anchors)
        assert c.line_start == min(s for s, _ in c.spans)


def test_corpus_chunk_sizes_in_target_band():
    docs = load_corpus([CORPUS], ["**/*.md"])
    chunks = [c for d in docs for c in chunk_document(d)]
    sizes = sorted(c.token_count for c in chunks)
    assert sizes[-1] <= 800
    median = sizes[len(sizes) // 2]
    assert 450 <= median <= 800, median


def test_citation_points_to_paragraph_anchor():
    docs = {d.doc_id: d for d in load_corpus([CORPUS], ["**/*.md"])}
    chunks = chunk_document(docs["tutorial/background-tasks.md"])
    c = chunks[0]
    i = next(i for i, p in enumerate(c.paragraphs) if p.startswith("Create a function to be run"))
    assert c.citation(c.spans[i]).startswith("tutorial/background-tasks.md#create-a-task-function (L")


def test_index_is_rebuilt_when_ingestion_code_changes(offline_settings, tmp_path, monkeypatch):
    import askmydocs.indexer as ix

    settings = dict(offline_settings)
    settings["index"] = {**offline_settings["index"], "dir": str(tmp_path / "idx")}
    from askmydocs.config import Settings

    settings = Settings(settings)
    builds = []
    monkeypatch.setattr(ix, "build_index", lambda s, log=print: builds.append(1) or ix.write_manifest(
        ix.index_paths(s), {"corpus_fingerprint": ix.corpus_fingerprint(
            [s.resolve_path(p) for p in s.corpus["paths"]], s.corpus["include"]),
            "chunking": dict(s.chunking), "pipeline_fingerprint": ix.pipeline_fingerprint(),
            "tokenizer": ix.tokenizer_name()}))
    ix.index_paths(settings).root.mkdir(parents=True)
    ix.ensure_index(settings, log=lambda *a: None)
    ix.ensure_index(settings, log=lambda *a: None)
    assert len(builds) == 1                      # fresh -> built once, then reused
    monkeypatch.setattr(ix, "pipeline_fingerprint", lambda: "different-code")
    ix.ensure_index(settings, log=lambda *a: None)
    assert len(builds) == 2                      # chunker code changed -> rebuilt
