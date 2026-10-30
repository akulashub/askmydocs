"""Document loaders. Each loader turns a file into a list of Blocks: the smallest
citable unit (a paragraph, list, code block or table) with its line/page span and
the heading it sits under. Chunks are assembled from blocks, so every citation can
point back to an exact paragraph."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Block:
    text: str
    line_start: int          # 1-based, inclusive (page number for PDFs)
    line_end: int
    section: str             # "Background Tasks > Add the background task"
    anchor: str = ""         # URL fragment of the nearest heading, if known
    kind: str = "text"       # text | code | heading


@dataclass
class Document:
    doc_id: str              # path relative to the corpus root, posix style
    title: str
    source_type: str         # markdown | pdf | html
    blocks: list[Block] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**asdict(self), "blocks": [asdict(b) for b in self.blocks]}

    @property
    def clean_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


# --------------------------------------------------------------------------- markdown

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w\-]+)\s*\})?\s*$")
_INCLUDE = re.compile(r"^\{\*\s*(\S+).*\*\}\s*$")          # FastAPI's {* file.py hl[1] *}
_ADMONITION_OPEN = re.compile(r"^/{3,4}\s*(\w+)(?:\s*\|\s*(.*))?$")
_ADMONITION_CLOSE = re.compile(r"^/{3,4}\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def _slugify(text: str) -> str:
    text = re.sub(r"[`*_]", "", text).lower()
    return re.sub(r"[^\w]+", "-", text).strip("-")


def load_markdown(path: Path, doc_id: str) -> Document:
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: list[tuple[int, str]] = []   # stack of (level, title)
    anchor = ""
    title = ""
    blocks: list[Block] = []
    buf: list[str] = []
    buf_start = 0
    in_fence = False
    pending_label = ""

    def section() -> str:
        return " > ".join(t for _, t in headings)

    def flush(end_line: int, kind: str = "text") -> None:
        nonlocal buf, pending_label
        text = "\n".join(buf).strip()
        if text:
            if pending_label and kind == "text":
                text = f"{pending_label}: {text}"
                pending_label = ""
            blocks.append(Block(text, buf_start, end_line, section(), anchor, kind))
        buf = []

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip()

        if in_fence:
            buf.append(line)
            if _FENCE.match(line):
                in_fence = False
                flush(i, "code")
            continue
        if _FENCE.match(line):
            flush(i - 1)
            in_fence, buf, buf_start = True, [line], i
            continue

        m = _HEADING.match(line)
        if m:
            flush(i - 1)
            level, text, explicit_anchor = len(m.group(1)), m.group(2).strip(), m.group(3)
            text = text.replace("`", "")
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, text))
            anchor = explicit_anchor or _slugify(text)
            if level == 1 and not title:
                title = text
            continue

        if _INCLUDE.match(line):
            # Code lives in separate files in this corpus; keep a pointer, not noise.
            flush(i - 1)
            continue
        if _ADMONITION_CLOSE.match(line):
            flush(i - 1)
            continue
        m = _ADMONITION_OPEN.match(line)
        if m:
            flush(i - 1)
            kind, label = m.group(1), (m.group(2) or "").strip()
            if kind != "tab":
                pending_label = kind.capitalize() + (f" ({label})" if label else "")
            continue

        if not line.strip():
            flush(i - 1)
            continue
        if not buf:
            buf_start = i
        buf.append(line)

    flush(len(lines), "code" if in_fence else "text")
    return Document(doc_id=doc_id, title=title or path.stem, source_type="markdown", blocks=blocks)


# --------------------------------------------------------------------------- pdf

def load_pdf(path: Path, doc_id: str) -> Document:
    from pypdf import PdfReader  # optional dependency: pip install .[loaders]

    reader = PdfReader(str(path))
    blocks: list[Block] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for para in re.split(r"\n\s*\n", text):
            para = re.sub(r"\s*\n\s*", " ", para).strip()
            if para:
                blocks.append(Block(para, page_no, page_no, f"Page {page_no}", f"page={page_no}"))
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem)
    return Document(doc_id=doc_id, title=title, source_type="pdf", blocks=blocks)


# --------------------------------------------------------------------------- html

def load_html(path: Path, doc_id: str) -> Document:
    from bs4 import BeautifulSoup  # optional dependency: pip install .[loaders]

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    headings: list[tuple[int, str]] = []
    anchor = ""
    blocks: list[Block] = []
    for n, el in enumerate(soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "table"]), start=1):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            level = int(el.name[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, text))
            anchor = el.get("id") or _slugify(text)
            continue
        kind = "code" if el.name == "pre" else "text"
        blocks.append(Block(text, n, n, " > ".join(t for _, t in headings), anchor, kind))
    title = soup.title.get_text(strip=True) if soup.title else path.stem
    return Document(doc_id=doc_id, title=title, source_type="html", blocks=blocks)


LOADERS = {".md": load_markdown, ".markdown": load_markdown, ".pdf": load_pdf,
           ".html": load_html, ".htm": load_html}


def load_corpus(roots: list[Path], include: list[str]) -> list[Document]:
    docs: list[Document] = []
    for root in roots:
        seen: set[Path] = set()
        for pattern in include:
            for path in sorted(root.glob(pattern)):
                if path in seen or path.suffix.lower() not in LOADERS:
                    continue
                seen.add(path)
                doc_id = path.relative_to(root).as_posix()
                docs.append(LOADERS[path.suffix.lower()](path, doc_id))
    return docs
