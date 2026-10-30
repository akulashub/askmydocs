"""HTTP API: `uvicorn askmydocs.api:app` (pip install .[api]).

POST /ask {"question": "..."} -> answer, refusal flag, sources with exact paragraphs.
GET  /healthz                  -> index manifest + active configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_settings
from .pipeline import RAGPipeline

app = FastAPI(title="Ask My Docs", version="0.1.0",
              description="Citation-enforced question answering over the indexed documentation.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class SourceOut(BaseModel):
    n: int
    citation: str
    paragraph: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str = ""
    sources: list[SourceOut]
    supported_fraction: float
    trace: dict


@lru_cache(maxsize=1)
def pipeline() -> RAGPipeline:
    return RAGPipeline(load_settings(profile=os.environ.get("ASKMYDOCS_PROFILE")))


@app.get("/healthz")
def healthz() -> dict:
    rag = pipeline()
    return {"status": "ok", "index": rag.manifest, "prompt": rag.prompt.ref,
            "llm": rag.llm.name, "reranker": rag.reranker.name}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        ans = pipeline().ask(req.question)
    except Exception as exc:  # surface backend errors (e.g. missing API key) clearly
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc
    trace = {k: v for k, v in ans.trace.items() if k != "candidates"}
    return AskResponse(answer=ans.answer, refused=ans.refused, refusal_reason=ans.refusal_reason,
                       sources=[SourceOut(n=s.n, citation=s.citation, paragraph=s.paragraph) for s in ans.sources],
                       supported_fraction=ans.supported_fraction, trace=trace)
