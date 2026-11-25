"""Deterministic sample selection. Records are dicts with at least `ref` and `keys`."""

import hashlib
import re
from collections import Counter, defaultdict

import numpy as np


def rng_for(seed: int, name: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def priority_order(records: list[dict], rng: np.random.Generator, split_priority: list[str]) -> list[int]:
    """Random order, then stable-sorted so preferred splits come first."""
    rank = {s: i for i, s in enumerate(split_priority)}
    perm = rng.permutation(len(records))
    return sorted(perm.tolist(), key=lambda i: rank.get(records[i]["ref"].get("split"), len(rank)))


def select_per_class(records: list[dict], n: int, rng, split_priority) -> list[int]:
    """Classification: up to n records per class."""
    taken: Counter = Counter()
    out = []
    for i in priority_order(records, rng, split_priority):
        k = records[i]["keys"][0]
        if taken[k] < n:
            taken[k] += 1
            out.append(i)
    return sorted(out)


def select_greedy_cover(records: list[dict], n: int, rng, split_priority) -> list[int]:
    """Detection/segmentation: every class appears in at least n selected images (when possible).

    Classes are visited rarest-first; images already selected count toward a class's quota.
    """
    order = priority_order(records, rng, split_priority)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i in order:
        for k in set(records[i]["keys"]):
            by_class[k].append(i)
    chosen: set[int] = set()
    have: Counter = Counter()
    for k in sorted(by_class, key=lambda k: (len(by_class[k]), k)):
        for i in by_class[k]:
            if have[k] >= n:
                break
            if i in chosen:
                continue
            chosen.add(i)
            for kk in set(records[i]["keys"]):
                have[kk] += 1
    return sorted(chosen)


def round_robin(cells: dict, per_cell: int, cap: int, rng) -> list[int]:
    """Take one item per cell per round until each cell has per_cell items or cap is hit."""
    names = list(cells)
    names = [names[i] for i in rng.permutation(len(names))]
    out = []
    for r in range(per_cell):
        for c in names:
            if len(out) >= cap:
                return sorted(out)
            if r < len(cells[c]):
                out.append(cells[c][r])
    return sorted(out)


QA_MC = re.compile(r"(?m)^\s*\(?[A-Ha-h][\.\):]\s+\S")


def qa_format(question: str, answer: str) -> str:
    if len(QA_MC.findall(question or "")) >= 2:
        return "multiple_choice"
    a = (answer or "").strip().lower().rstrip(".")
    if a in ("yes", "no"):
        return "yes_no"
    if re.fullmatch(r"-?\d+(\.\d+)?%?", a):
        return "numeric"
    return "open_ended"


def choose_strata(records: list[dict], cfg_itt: dict, forced: list[str] | None) -> list[str]:
    """config (if several) + one primary key + one secondary key + qa_format, while cells fit the cap."""
    if forced:
        return forced
    n = len(records)
    max_cells = max(1, cfg_itt["max_per_dataset"] // cfg_itt["per_stratum"])

    def usable(key):
        vals = [r["strata"].get(key) for r in records]
        present = sum(v is not None for v in vals)
        distinct = len(set(v for v in vals if v is not None))
        return present >= 0.9 * n and 2 <= distinct <= cfg_itt["max_stratum_values"]

    def cells(keys):
        return len({tuple(r["strata"].get(k) for k in keys) for r in records})

    def adds_info(keys, k):
        return cells(keys + [k]) > cells(keys) if keys else True

    keys = ["config"] if usable("config") else []
    primary = next((k for k in cfg_itt["primary_keys"] + ["id_prefix"] if usable(k) and adds_info(keys, k)), None)
    if primary:
        keys.append(primary)
    for k in cfg_itt.get("secondary_keys", []):
        if k not in keys and usable(k) and adds_info(keys, k) and cells(keys + [k]) <= max_cells:
            keys.append(k)
            break
    if usable("qa_format") and adds_info(keys, "qa_format") and cells(keys + ["qa_format"]) <= max_cells:
        keys.append("qa_format")
    return keys


def select_itt(records: list[dict], cfg_itt: dict, rng, split_priority, forced: list[str] | None = None) -> tuple[list[int], list[str]]:
    """Stratified by metadata keys; falls back to TF-IDF k-means over question text."""
    per, cap = cfg_itt["per_stratum"], cfg_itt["max_per_dataset"]
    keys = choose_strata(records, cfg_itt, forced)
    order = priority_order(records, rng, split_priority)
    cells: dict = defaultdict(list)
    if keys:
        for i in order:
            cells[tuple(str(records[i]["strata"].get(k)) for k in keys)].append(i)
        return round_robin(cells, _quota(per, cap, len(cells), cfg_itt), cap, rng), keys

    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [records[i]["question"] or "" for i in order]
    k = max(1, min(cap // per, len(texts)))
    x = TfidfVectorizer(max_features=20000, stop_words="english").fit_transform(texts)
    km = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(1 << 31))).fit(x)
    dist = km.transform(x)
    for pos, lab in enumerate(km.labels_):
        cells[int(lab)].append((dist[pos, lab], order[pos]))
    cells = {c: [i for _, i in sorted(v)] for c, v in cells.items()}
    for pos, lab in enumerate(km.labels_):
        records[order[pos]]["strata"]["cluster"] = int(lab)
    return round_robin(cells, _quota(per, cap, len(cells), cfg_itt), cap, rng), ["cluster"]


def _quota(per: int, cap: int, n_cells: int, cfg_itt: dict) -> int:
    """per_stratum is the minimum; with fill_to_cap, spread the cap evenly over strata."""
    if cfg_itt.get("fill_to_cap"):
        return max(per, cap // max(1, n_cells))
    return per
