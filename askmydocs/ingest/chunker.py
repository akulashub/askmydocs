"""Structure-aware, token-budgeted chunking with overlap.

Strategy
- Work on "units": a block, or a sentence slice of a block that is too big.
- Pack units into chunks up to `target_tokens` (hard cap `max_tokens`).
- Prefer to break at H2 headings once a chunk has `section_break_tokens`, so
  chunks rarely straddle two topics but still land in the 500-800 token band.
- Seed each new chunk with the trailing units of the previous chunk, up to
  `overlap_tokens`, so a sentence at a boundary is always seen with its context.
- Every chunk keeps the line spans of the units it contains, which is what lets a
  citation point to an exact paragraph instead of "somewhere in this file".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

from ..tokens import count_tokens
from .loaders import Block, Document

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*\[(])")


@dataclass
class Unit:
    text: str
    line_start: int
    line_end: int
    section: str
    anchor: str
    tokens: int


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    anchor: str
    text: str
    line_start: int
    line_end: int
    token_count: int
    spans: list[tuple[int, int]] = field(default_factory=list)  # line span of each paragraph
    paragraphs: list[str] = field(default_factory=list)         # aligned with `spans`
    anchors: list[str] = field(default_factory=list)            # heading anchor per paragraph
    position: int = 0

    @property
    def embed_text(self) -> str:
        """Breadcrumb-prefixed text: improves retrieval of short, context-poor chunks."""
        return f"{self.title} > {self.section}\n\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spans"] = [list(s) for s in self.spans]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        d = dict(d)
        d["spans"] = [tuple(s) for s in d.get("spans", [])]
        return cls(**d)

    def citation(self, span: tuple[int, int] | None = None) -> str:
        s, e = span or (self.line_start, self.line_end)
        anchor = self.anchor
        if span is not None and span in self.spans and self.anchors:
            anchor = self.anchors[self.spans.index(span)]
        loc = f"L{s}" if s == e else f"L{s}-L{e}"
        return f"{self.doc_id}#{anchor} ({loc})" if anchor else f"{self.doc_id} ({loc})"


def _split_block(block: Block, max_unit_tokens: int) -> list[Unit]:
    toks = count_tokens(block.text)
    if toks <= max_unit_tokens or block.kind == "code":
        if block.kind == "code" and toks > max_unit_tokens:
            return _window(block, max_unit_tokens)
        return [Unit(block.text, block.line_start, block.line_end, block.section, block.anchor, toks)]
    units: list[Unit] = []
    for sent in _SENT_SPLIT.split(block.text):
        sent = sent.strip()
        if not sent:
            continue
        st = count_tokens(sent)
        if st > max_unit_tokens:
            units.extend(_window(Block(sent, block.line_start, block.line_end, block.section,
                                       block.anchor, block.kind), max_unit_tokens))
        else:
            units.append(Unit(sent, block.line_start, block.line_end, block.section, block.anchor, st))
    return units


def _window(block: Block, size: int) -> list[Unit]:
    """Last resort for giant code blocks / run-on text: split on lines, then words."""
    out: list[Unit] = []
    cur: list[str] = []
    for piece in re.split(r"(\n)", block.text) if "\n" in block.text else block.text.split(" "):
        candidate = ("" if "\n" in block.text else " ").join(cur + [piece])
        if cur and count_tokens(candidate) > size:
            text = ("" if "\n" in block.text else " ").join(cur).strip()
            if text:
                out.append(Unit(text, block.line_start, block.line_end, block.section, block.anchor,
                                count_tokens(text)))
            cur = [piece]
        else:
            cur.append(piece)
    text = ("" if "\n" in block.text else " ").join(cur).strip()
    if text:
        out.append(Unit(text, block.line_start, block.line_end, block.section, block.anchor,
                        count_tokens(text)))
    return out


def chunk_document(
    doc: Document,
    target_tokens: int = 600,
    max_tokens: int = 800,
    min_tokens: int = 120,
    overlap_tokens: int = 100,
    section_break_tokens: int = 450,
) -> list[Chunk]:
    if not doc.blocks:
        return []
    units: list[Unit] = []
    for b in doc.blocks:
        # Units no bigger than the overlap budget guarantee the overlap is never empty.
        units.extend(_split_block(b, max_unit_tokens=overlap_tokens))

    groups: list[list[Unit]] = []
    current: list[Unit] = []
    current_tokens = 0
    new_units_in_current = 0

    def emit() -> None:
        nonlocal current, current_tokens, new_units_in_current
        if new_units_in_current:
            groups.append(current)
        # seed next chunk with overlap from the tail of this one
        tail: list[Unit] = []
        t = 0
        for u in reversed(current):
            if t + u.tokens > overlap_tokens:
                break
            tail.insert(0, u)
            t += u.tokens
        current, current_tokens, new_units_in_current = tail, t, 0

    prev_section = units[0].section
    for u in units:
        section_changed = u.section != prev_section
        prev_section = u.section
        would_be = current_tokens + u.tokens
        if new_units_in_current and (
            would_be > max_tokens
            or (would_be > target_tokens and current_tokens >= min_tokens)
            or (section_changed and current_tokens >= section_break_tokens and _top(u.section) != _top_of(current))
        ):
            emit()
            if section_changed:
                # Don't carry overlap across a topic boundary: it would mix sections.
                current, current_tokens = [], 0
        current.append(u)
        current_tokens += u.tokens
        new_units_in_current += 1
    if new_units_in_current:
        groups.append(current)

    # Merge a tiny final chunk into its predecessor when the result still fits.
    if len(groups) >= 2:
        last_new = sum(u.tokens for u in groups[-1])
        if last_new < min_tokens and sum(u.tokens for u in groups[-2]) + last_new <= max_tokens:
            tail = [u for u in groups[-1] if u not in groups[-2]]
            groups[-2] = groups[-2] + tail
            groups.pop()

    chunks: list[Chunk] = []
    for pos, group in enumerate(groups):
        paragraphs: list[str] = []
        anchors: list[str] = []
        spans: list[tuple[int, int]] = []
        for i, u in enumerate(group):
            span = (u.line_start, u.line_end)
            if i > 0 and spans[-1] == span:
                paragraphs[-1] += " " + u.text      # sentence slices of the same paragraph
            else:
                paragraphs.append(u.text)
                anchors.append(u.anchor)
                spans.append(span)
        text = "\n\n".join(paragraphs)
        first = next((u for u in group if u.section), group[0])
        digest = hashlib.sha1(f"{doc.doc_id}:{pos}:{text}".encode()).hexdigest()[:10]
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}::{pos:03d}::{digest}",
            doc_id=doc.doc_id,
            title=doc.title,
            section=first.section,
            anchor=first.anchor,
            text=text,
            line_start=min(s for s, _ in spans),
            line_end=max(e for _, e in spans),
            token_count=count_tokens(text),
            spans=spans,
            paragraphs=paragraphs,
            anchors=anchors,
            position=pos,
        ))
    return chunks


def _top(section: str) -> str:
    """Section path truncated to H1 > H2: the level at which we prefer to break."""
    return " > ".join(section.split(" > ")[:2])


def _top_of(units: list[Unit]) -> str:
    return _top(units[-1].section) if units else ""


def chunk_corpus(docs: list[Document], **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, **kwargs))
    return out
