"""Versioned prompt registry backed by config/prompts.yaml."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .config import ROOT

PROMPTS_FILE = ROOT / "config" / "prompts.yaml"
LOCK_FILE = ROOT / "config" / "prompts.lock"


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    fields: dict

    @property
    def content_hash(self) -> str:
        body = {k: v for k, v in self.fields.items() if k not in {"version", "description"}}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]

    @property
    def ref(self) -> str:
        """Short identifier recorded on answers and eval reports."""
        return f"{self.id}@{self.version}#{self.content_hash}"

    def render(self, part: str, **kwargs) -> str:
        return self.fields[part].format(**kwargs)

    def __getitem__(self, key: str):
        return self.fields[key]


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def get_prompt(prompt_id: str, path: Path | None = None) -> Prompt:
    data = _load(str(path or PROMPTS_FILE))
    fields = data["prompts"][prompt_id]
    return Prompt(id=prompt_id, version=str(fields["version"]), fields=fields)


def all_prompts(path: Path | None = None) -> list[Prompt]:
    data = _load(str(path or PROMPTS_FILE))
    return [get_prompt(pid, path) for pid in data["prompts"]]


def lock_snapshot(path: Path | None = None) -> dict:
    return {p.id: {"version": p.version, "hash": p.content_hash} for p in all_prompts(path)}
