"""Resolve the crop named in a prompt from dataset metadata."""

import re

from agbench.eval.resources import norm

MAX_LISTED = 4


def catalog_crops(entry: dict) -> list[str]:
    crops = entry.get("crops") or []
    crops = [crops] if isinstance(crops, str) else crops
    return [str(c).strip() for c in crops if c and not str(c).startswith("+") and str(c).lower() not in ("unknown", "none", "n/a")]


def _short(crop: str) -> str:
    """'tulsi (ocimum tenuiflorum)' -> 'tulsi'."""
    return re.sub(r"\s*\(.*?\)", "", crop).strip()


def resolve(entry: dict, bucket: str, labels: list[str] = (), sample: dict | None = None) -> tuple[str, str]:
    """(crop, source). Species ID always uses 'plant' to avoid naming the answer."""
    if bucket == "species_id":
        return "plant", "species_id_plant"
    meta = (sample or {}).get("strata") or {}
    if meta.get("crop") and str(meta["crop"]).lower() != "unknown":
        return str(meta["crop"]).lower(), "sample_metadata"
    crops = catalog_crops(entry)
    if len(crops) == 1:
        return _short(crops[0]), "catalog_single"
    if not crops:
        return "plant", "none"
    text = " " + " ".join(norm(l) for l in labels) + " "
    hits = [c for c in crops if norm(_short(c)) and f" {norm(_short(c))} " in text]
    if hits:
        return _short(max(hits, key=len)), "label_match"
    names = [_short(c) for c in crops[:MAX_LISTED]]
    return (", ".join(names[:-1]) + " or " + names[-1]), "catalog_list"


def phrase(crop: str) -> str:
    """'tomato' -> 'a tomato plant'; 'plant' -> 'a plant'."""
    p = crop if crop == "plant" or crop.endswith(" plant") else f"{crop} plant"
    return ("an " if p[0].lower() in "aeiou" else "a ") + p
