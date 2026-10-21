"""Packaged YAML resources (prompts, models, ag tasks)."""

import os
import re
from functools import cache
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


@cache
def load(name: str) -> dict:
    """Load a packaged YAML; AGBENCH_<NAME> env var points to a replacement file."""
    override = os.environ.get("AGBENCH_" + Path(name).stem.upper())
    return yaml.safe_load(Path(override or HERE / name).read_text())


def prompts() -> dict:
    return load("prompts.yaml")


def models() -> dict:
    return load("models.yaml")


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
