"""Frozen, named benchmark versions ("bundles"), e.g. agml-full-v1.

A bundle fixes everything that decides a score: which images (manifests: dataset, HF repo, pinned revision, sample
refs), the evaluation items built from them (prompt, options, answer, ground truth), and the judge prompt. Images are
not redistributed: `agbench data pull <bundle>` downloads exactly the listed samples from the source AgML repos at
their pinned revisions.

    benchmarks/<name>/bundle.yaml        name, description, counts, judge, sha256 of each file, hash
    benchmarks/<name>/items.jsonl.gz     frozen EvalItems (image paths relative to the data dir)
    benchmarks/<name>/manifests.json.gz  {dataset: manifest + the catalog fields needed to download it}
    benchmarks/<name>/config.yaml        sampling config the manifests were drawn with (overrides used when extracting)
"""

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

BUNDLE_DIR = Path(__file__).parent / "benchmarks"
DEFAULT = "agml-full-v1"
CATALOG_KEYS = ("name", "hf_repo", "task", "agricultural_task", "classes", "license", "citation", "crops", "sensor_modality",
                "real_or_synthetic", "eligible")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_gz(path: Path, text: str) -> None:
    """Byte-reproducible gzip (no timestamp or file name in the header)."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(text.encode())


def judge_prompt_sha() -> str:
    from agbench.eval.resources import prompts

    p = prompts()
    return _sha(json.dumps([p["judge_system"], p["judge_classification"], p["judge_qa"]]).encode())[:12]


@dataclass
class Bundle:
    name: str
    path: Path
    spec: dict = field(repr=False)

    @property
    def hash(self) -> str:
        return self.spec["hash"]

    def _text(self, fname: str) -> str:
        return gzip.decompress((self.path / fname).read_bytes()).decode()

    @cached_property
    def manifests(self) -> dict[str, dict]:
        return json.loads(self._text("manifests.json.gz"))

    @cached_property
    def raw_items(self) -> list[dict]:
        return [json.loads(l) for l in self._text("items.jsonl.gz").splitlines()]

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    def check(self) -> list[str]:
        """Problems with the bundle files themselves (edited or corrupted after freezing)."""
        out = []
        for fname, want in self.spec["files"].items():
            p = self.path / fname
            if not p.exists():
                out.append(f"{fname}: missing")
            elif _sha(p.read_bytes()) != want:
                out.append(f"{fname}: sha256 differs from bundle.yaml")
        if _bundle_hash(self.spec) != self.hash:
            out.append("bundle.yaml: hash does not match its contents")
        return out

    def datasets(self, task: list[str] | None = None, only: str | None = None) -> list[str]:
        from agbench.pipeline import TASK_DIRS

        return [n for n, m in sorted(self.manifests.items())
                if (not task or TASK_DIRS[m["task"]] in task or m["task"] in task) and (not only or re.search(only, n))]

    def entry(self, name: str) -> dict:
        """Catalog-style entry for the download pipeline."""
        return self.manifests[name]["catalog"]

    def items(self, data_dir, tasks=None, formats=None, only: str | None = None, limit_per_dataset: int | None = None,
              seed: int = 0) -> tuple[list, list[dict]]:
        """Frozen EvalItems for the datasets present in data_dir, plus the selected datasets that are not."""
        from agbench.eval.items import EvalItem, _unit
        from agbench.pipeline import base_dir

        data_dir = Path(data_dir)
        present = {}
        for r in self.raw_items:
            key = (r["task"], r["dataset"])
            if key not in present:
                present[key] = (base_dir(data_dir) / r["task"] / r["dataset"] / "meta.json").exists()
        keep, skipped, seen = [], [], set()
        for r in self.raw_items:
            if tasks and r["task"] not in tasks:
                continue
            if formats and r["format"] not in formats and not (r["task"] == "image_text_to_text" and r["qa_format"] in formats):
                continue
            if only and not re.search(only, r["dataset"]):
                continue
            key = (r["task"], r["dataset"])
            if not present[key]:
                if key not in seen:
                    seen.add(key)
                    skipped.append({"dataset": r["dataset"], "task": r["task"], "reason": f"not downloaded (agbench data pull {self.name})"})
                continue
            keep.append(r)
        if limit_per_dataset:
            by_ds: dict = {}
            for r in keep:
                by_ds.setdefault(r["dataset"], set()).add(r["sample_id"])
            chosen = {s for ids in by_ds.values() for s in sorted(ids, key=lambda s: _unit(seed, s))[:limit_per_dataset]}
            keep = [r for r in keep if r["sample_id"] in chosen]
        items = []
        for r in keep:
            it = EvalItem(**r)
            it.images = [str(data_dir / p) for p in r["images"]]
            items.append(it)
        return items, skipped

    def summary(self) -> dict:
        return {k: self.spec[k] for k in ("name", "description", "hash", "created", "datasets", "items", "judge")}


def _bundle_hash(spec: dict) -> str:
    payload = {"name": spec["name"], "files": spec["files"], "judge": spec["judge"]}
    return _sha(json.dumps(payload, sort_keys=True).encode())[:16]


def list_bundles() -> list[str]:
    return sorted(p.name for p in BUNDLE_DIR.iterdir() if (p / "bundle.yaml").exists()) if BUNDLE_DIR.is_dir() else []


def load(name: str) -> Bundle:
    p = BUNDLE_DIR / name
    if not (p / "bundle.yaml").exists():
        raise KeyError(f"unknown benchmark {name!r}; available: {', '.join(list_bundles()) or 'none'}")
    return Bundle(name, p, yaml.safe_load((p / "bundle.yaml").read_text()))


def freeze(name: str, items_path: Path, description: str, manifest_dir: Path | None = None, config_path: Path | None = None,
           catalog: dict | None = None, judge_model: str | None = None) -> Bundle:
    """Write benchmarks/<name>/ from an items.jsonl (the items a run evaluated) and the committed manifests.

    Items are stored exactly as given (image paths made relative to the data dir). Only datasets with at least one
    item are included; a manifest that yields no items (e.g. no usable segmentation classes) is left out."""
    from datetime import datetime, timezone

    from agbench.catalog import load_catalog
    from agbench.config import DEFAULT_CONFIG, MANIFEST_DIR
    from agbench.eval.resources import models
    from agbench.pipeline import TASK_DIRS

    manifest_dir, config_path = Path(manifest_dir or MANIFEST_DIR), Path(config_path or DEFAULT_CONFIG)
    catalog = catalog or load_catalog()
    out = BUNDLE_DIR / name
    if out.exists():
        raise FileExistsError(f"{out} exists: a frozen benchmark is never overwritten; freeze a new name")

    items = []
    for line in open(items_path):
        r = json.loads(line)
        r["images"] = [p[p.index("base-data/"):] for p in r["images"]]
        items.append(r)
    used = {r["dataset"] for r in items}
    manifests = {}
    for p in sorted(manifest_dir.glob("*/*.json")):
        m = json.loads(p.read_text())
        if m["name"] in used:
            e = catalog["datasets"][m["name"]]
            m["catalog"] = {k: e.get(k) for k in CATALOG_KEYS}
            manifests[m["name"]] = m
    missing = used - set(manifests)
    if missing:
        raise ValueError(f"items from datasets without a manifest: {sorted(missing)[:5]}")

    out.mkdir(parents=True)
    _write_gz(out / "items.jsonl.gz", "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in items))
    _write_gz(out / "manifests.json.gz", json.dumps(manifests, ensure_ascii=False, sort_keys=True, default=str))
    (out / "config.yaml").write_bytes(config_path.read_bytes())
    counts: dict = {}
    for r in items:
        k = f"{r['task']}/{r['format'] if r['task'] != 'image_text_to_text' else r['qa_format']}"
        counts[k] = counts.get(k, 0) + 1
    by_task: dict = {}
    for m in manifests.values():
        by_task[TASK_DIRS[m["task"]]] = by_task.get(TASK_DIRS[m["task"]], 0) + 1
    spec = {
        "name": name, "description": description,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {"total": len(manifests), **dict(sorted(by_task.items()))},
        "images": sum(len(m["samples"]) for m in manifests.values()),
        "items": {"total": len(items), **dict(sorted(counts.items()))},
        "judge": {"model": judge_model or models()["judge"], "prompt_sha": judge_prompt_sha()},
        "files": {f: _sha((out / f).read_bytes()) for f in ("items.jsonl.gz", "manifests.json.gz", "config.yaml")},
    }
    spec["hash"] = _bundle_hash(spec)
    (out / "bundle.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    return load(name)


# ---------------------------------------------------------------- local data

def data_status(b: Bundle, data_dir, task=None, only=None) -> dict[str, str]:
    """Per dataset: done | missing | stale (wrong revision or different samples) | failed."""
    from agbench.pipeline import base_dir, failure_path, TASK_DIRS

    out = {}
    for n in b.datasets(task, only):
        m = b.manifests[n]
        d = base_dir(data_dir) / TASK_DIRS[m["task"]] / n
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            have = {json.loads(l)["sample_id"] for l in open(d / "samples.jsonl")} if (d / "samples.jsonl").exists() else set()
            same = meta.get("hf_sha") == m["hf_sha"] and have == {s["sample_id"] for s in m["samples"]}
            out[n] = "done" if same else "stale"
        else:
            out[n] = "failed" if failure_path(Path(data_dir), n).exists() else "missing"
    return out


def verify(b: Bundle, data_dir, task=None, only=None, decode: bool = True) -> dict[str, list[str]]:
    """Problems per dataset in a local copy: not downloaded, wrong revision, missing or extra samples, missing or
    undecodable images. An empty dict means the local copy matches the bundle."""
    from PIL import Image, ImageFile

    from agbench.pipeline import base_dir, TASK_DIRS

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # as the model servers and YOLO decode them; only unreadable files count
    need: dict[str, set] = {}
    for r in b.raw_items:
        need.setdefault(r["dataset"], set()).update(r["images"])
    problems: dict[str, list[str]] = {}
    for n in b.datasets(task, only):
        m, errs = b.manifests[n], []
        d = base_dir(data_dir) / TASK_DIRS[m["task"]] / n
        if not (d / "meta.json").exists():
            problems[n] = ["not downloaded"]
            continue
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("hf_sha") != m["hf_sha"]:
            errs.append(f"revision {str(meta.get('hf_sha'))[:7]} != {m['hf_sha'][:7]}")
        have = {json.loads(l)["sample_id"] for l in open(d / "samples.jsonl")}
        want = {s["sample_id"] for s in m["samples"]}
        if want - have:
            errs.append(f"{len(want - have)} samples missing")
        if have - want:
            errs.append(f"{len(have - want)} samples not in the benchmark")
        bad = 0
        for rel in sorted(need.get(n, ())):
            p = Path(data_dir) / rel
            if not p.exists():
                errs.append(f"missing image {rel}")
                continue
            if decode:
                try:
                    with Image.open(p) as im:
                        im.load()
                except Exception:
                    bad += 1
        if bad:
            errs.append(f"{bad} images do not decode")
        if errs:
            problems[n] = errs
    return problems


# ---------------------------------------------------------------- re-pinning a recreated source repo

def repin(b: Bundle, name: str, reference_dir, write: bool = False, scratch=None) -> dict:
    """Point a VQA dataset's manifest at the current revision of its HF repo, for a repo that was deleted and
    re-uploaded (the pinned revision no longer exists).

    The frozen items hold every question, option and answer, so the source only has to supply the images. Each sample
    is found in the new upload by the SHA-256 of its images in the reference copy (a data dir holding the benchmark's
    images; uploads name image files by that hash), preferring a row with the same question when several rows share
    the image. The chosen rows are then downloaded and must give byte-identical images. Sample ids, and so the
    frozen items, are unchanged. Rows whose conversation text differs from the reference are counted (`text_changed`):
    they do not affect frozen-item scores. With `write`, the bundle and manifests/base-data/ are updated only if
    every sample's images matched."""
    import hashlib
    import shutil
    import tempfile
    from datetime import datetime, timezone

    from agbench import build
    from agbench.config import MANIFEST_DIR, load_config
    from agbench.hub import Repo
    from agbench.pipeline import TASK_DIRS, base_dir, index_dataset
    from agbench.sampling import rng_for

    m = b.manifests[name]
    if m["task"] != "image_text_to_text":
        raise NotImplementedError("re-pinning matches images by content hash, which only VQA uploads use as file names")
    cfg = load_config(b.config_path)
    ref_dir = base_dir(reference_dir) / TASK_DIRS[m["task"]] / name
    ref = {s["sample_id"]: s for s in (json.loads(l) for l in open(ref_dir / "samples.jsonl"))}
    repo = Repo(m["hf_repo"], revision=None)
    recs, info = index_dataset(cfg, m["catalog"], repo, rng_for(cfg["sampling"]["seed"], name))
    keys: dict = {}
    for f in sorted({r["ref"]["file"] for r in recs}):
        for i, x in enumerate(repo.parquet(f).read(columns=["images"]).to_pylist()):
            keys[(f, i)] = tuple(im.get("path") for im in (x["images"] or []))
    by_key: dict = {}
    for r in recs:
        by_key.setdefault(keys.get((r["ref"]["file"], r["ref"]["row"])), []).append(r)

    report = {"dataset": name, "old_revision": m["hf_sha"], "new_revision": repo.revision, "samples": len(m["samples"]),
              "unmatched": [], "different": [], "text_changed": 0}
    new_samples, picks = [], []
    for s in m["samples"]:
        rs = ref.get(s["sample_id"])
        if not rs:
            report["unmatched"].append(s["sample_id"])
            continue
        key = tuple(hashlib.sha256((ref_dir / p).read_bytes()).hexdigest() for p in rs["images"])
        q = (build._qa(rs["messages"])[0] or "")[:500]
        cands = by_key.get(key, [])
        if not cands:
            report["unmatched"].append(s["sample_id"])
            continue
        pick = next((r for r in cands if r["question"] == q), cands[0])
        picks.append({**pick, "sample_id": s["sample_id"]})
        new_samples.append({**s, "ref": pick["ref"]})

    tmp = Path(tempfile.mkdtemp(prefix=f"repin_{name}_", dir=scratch))
    try:
        got = build.extract(repo, name, m["task"], info, picks, tmp, None)
        for g in got:
            r0 = ref[g["sample_id"]]
            same_imgs = len(g["images"]) == len(r0["images"]) and all(
                (tmp / a).read_bytes() == (ref_dir / c).read_bytes() for a, c in zip(g["images"], r0["images"]))
            if not same_imgs:
                report["different"].append(g["sample_id"])
            elif g["messages"] != r0["messages"]:
                report["text_changed"] += 1
        report["different"] += sorted({p["sample_id"] for p in picks} - {g["sample_id"] for g in got})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report["ok"] = not (report["unmatched"] or report["different"])
    if write and report["ok"]:
        m.update(hf_sha=repo.revision, samples=new_samples, info={**m["info"], "layout": info["layout"]})
        _write_gz(b.path / "manifests.json.gz", json.dumps(b.manifests, ensure_ascii=False, sort_keys=True, default=str))
        spec = b.spec
        spec["files"]["manifests.json.gz"] = _sha((b.path / "manifests.json.gz").read_bytes())
        spec.setdefault("repinned", []).append({
            "dataset": name, "from": report["old_revision"], "to": repo.revision,
            "date": datetime.now(timezone.utc).date().isoformat(),
            "reason": f"source repo re-uploaded; images matched by SHA-256, byte-identical; "
                      f"{report['text_changed']} of {report['samples']} rows have different conversation text "
                      f"(frozen items unaffected)"})
        spec["hash"] = _bundle_hash(spec)
        (b.path / "bundle.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
        mp = MANIFEST_DIR / TASK_DIRS[m["task"]] / f"{name}.json"
        if mp.exists():
            committed = json.loads(mp.read_text())
            committed.update(hf_sha=repo.revision, samples=new_samples)
            mp.write_text(json.dumps(committed, indent=1, ensure_ascii=False, default=str))
        report["written"] = True
    return report

