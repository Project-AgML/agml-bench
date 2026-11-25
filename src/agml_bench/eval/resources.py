"""Packaged YAML resources (prompts, models, ag tasks)."""

import os
import re
from functools import cache
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


@cache
def load(name: str) -> dict:
    """Load a packaged YAML; AGML_BENCH_<NAME> env var points to a replacement file."""
    override = os.environ.get("AGML_BENCH_" + Path(name).stem.upper())
    return yaml.safe_load(Path(override or HERE / name).read_text())


def prompts() -> dict:
    return load("prompts.yaml")


def models() -> dict:
    reg = load("models.yaml")
    for p in filter(None, os.environ.get("AGML_BENCH_MODELS_EXTRA", "").split(os.pathsep)):
        if p not in reg.setdefault("_extra", []):
            register_models(p)
    return reg


def register_models(src) -> None:
    """Add or override registry models from a YAML path or a dict.

    Accepts the models.yaml layout ({models: {...}, defaults: {...}, judge: ...}) or a bare {model_id: spec}.
    A new model without `backend` is served with vLLM; for a registered model only the given fields change."""
    reg = load("models.yaml")
    if not isinstance(src, dict):
        reg.setdefault("_extra", []).append(str(src))
        src = yaml.safe_load(Path(src).read_text()) or {}
    if not {"models", "defaults", "judge"} & set(src):
        src = {"models": src}
    reg["defaults"] = {**reg["defaults"], **(src.get("defaults") or {})}
    reg["judge"] = src.get("judge", reg["judge"])
    for mid, spec in (src.get("models") or {}).items():
        reg["models"][mid] = {**reg["models"].get(mid, {"backend": "vllm"}), **(spec or {})}  # override only the given fields


def ag_tasks() -> dict:
    return load("ag_tasks.yaml")


def seg_classes() -> dict:
    return load("seg_classes.yaml")


def norm(s) -> str:
    """Lowercase, underscores/punctuation to spaces, collapsed whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()


def pretty(label: str) -> str:
    """Display form of a class label: underscores to spaces."""
    return re.sub(r"\s+", " ", re.sub(r"_+", " ", str(label))).strip()
