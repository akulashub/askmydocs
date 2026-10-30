"""Dataset and prompt integrity: these run on every PR before the eval itself."""

import json
import re

from askmydocs.config import ROOT
from askmydocs.ingest.loaders import load_corpus
from askmydocs.prompts import LOCK_FILE, all_prompts, get_prompt, lock_snapshot

GOLDEN = ROOT / "eval" / "golden.jsonl"
CORPUS = ROOT / "data" / "corpus" / "fastapi"


def _norm(s):
    return " ".join(s.split())


def _rows():
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def test_golden_set_size_and_schema():
    rows = _rows()
    assert 50 <= len(rows) <= 200
    assert len({r["id"] for r in rows}) == len(rows)
    assert sum(not r["answerable"] for r in rows) >= 10
    for r in rows:
        assert r["question"].strip().endswith("?")
        if r["answerable"]:
            assert r["reference_answer"] and r["sources"]


def test_every_evidence_span_exists_verbatim_in_its_document():
    docs = {d.doc_id: _norm(d.clean_text) for d in load_corpus([CORPUS], ["**/*.md"])}
    for r in _rows():
        for s in r["sources"]:
            assert s["doc_id"] in docs, (r["id"], s["doc_id"])
            assert _norm(s["evidence"]) in docs[s["doc_id"]], (r["id"], s["evidence"][:60])


def test_unanswerable_items_are_really_unanswerable():
    text = " ".join(_norm(d.clean_text).lower() for d in load_corpus([CORPUS], ["**/*.md"]))
    for r in _rows():
        for term in r.get("absent_terms", []):
            assert not re.search(r"\b" + re.escape(term) + r"\b", text), (r["id"], term)


def test_prompt_changes_require_version_bump():
    """Editing a prompt without bumping its version fails CI. After bumping, run
    `askmydocs prompts --lock` and commit config/prompts.lock."""
    locked = json.loads(LOCK_FILE.read_text())
    for pid, cur in lock_snapshot().items():
        assert pid in locked, f"new prompt {pid}: run `askmydocs prompts --lock`"
        if cur["hash"] != locked[pid]["hash"]:
            assert cur["version"] != locked[pid]["version"], f"prompt {pid} changed without a version bump"
            raise AssertionError(f"prompt {pid} bumped to {cur['version']}: run `askmydocs prompts --lock`")


def test_prompt_templates_render():
    p = get_prompt("answer_with_citations")
    sys_ = p.render("system", domain="X", refusal_token=p["refusal_token"])
    user = p.render("user", question="Q?", context="[1] ...", refusal_token=p["refusal_token"])
    assert p["refusal_token"] in sys_ and "Q?" in user
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", x.version) for x in all_prompts())
