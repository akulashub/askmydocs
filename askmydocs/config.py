"""Settings loading: YAML defaults -> profile overrides -> environment overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"
ENV_PREFIX = "ASKMYDOCS__"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw


class Settings(dict):
    """A dict with attribute-style section access: settings.retrieval['vector_k']."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def resolve_path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p


def load_settings(
    path: str | Path | None = None,
    profile: str | None = None,
    overrides: dict | None = None,
) -> Settings:
    path = Path(path) if path else DEFAULT_SETTINGS
    data = yaml.safe_load(path.read_text())
    profiles = data.pop("profiles", {}) or {}

    profile = profile or os.environ.get("ASKMYDOCS_PROFILE", "default")
    if profile not in profiles and profile != "default":
        raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(profiles)}")
    data = _deep_merge(data, profiles.get(profile, {}) or {})

    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        parts = env_key[len(ENV_PREFIX):].lower().split("__")
        if len(parts) != 2:
            continue
        section, key = parts
        if section in data and isinstance(data[section], dict):
            data[section][key] = _coerce(raw, data[section].get(key, ""))

    if overrides:
        data = _deep_merge(data, overrides)
    data["profile"] = profile
    return Settings(data)
