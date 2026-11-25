"""Build evaluation items from sampled base-data (samples.jsonl + meta.json)."""

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from agml_bench.eval import agtask, crops
from agml_bench.eval.resources import norm, pretty, prompts, seg_classes
from agml_bench.eval.scoring import parse_number, parse_yes_no, similarity

TASKS = ["classification", "detection", "segmentation", "image_text_to_text"]
FORMATS = {
    "classification": ["mcqa_within", "mcqa_across", "oeq"],
    "detection": ["grounding"],
    "segmentation": ["grounding"],
    "image_text_to_text": ["native", "multiple_choice"],  # native = every ITT question; multiple_choice = only those
}
N_OPTIONS = 4
NOTA_SHARE = 0.3
MIN_COMPONENT_FRAC = 0.0002  # mask components smaller than 0.02% of the image are ignored
NON_CLASSES = re.compile(r"^(background|value_\d+|unlabeled|ignore|void|__many__|[\d,\s]+)$", re.I)


@dataclass
class EvalItem:
    item_id: str
    sample_id: str
    dataset: str
    task: str
    format: str
    images: list[str]
    prompt: str
    ag_task: str
    crop: str
    crop_source: str
    choices: list[str] | None = None      # options as shown (NOTA/IDK last unless nota_position="random")
    answer: object = None                 # correct option text / letters / yes-no / number / reference text
    gt_label: str | None = None
    gt_boxes: list[dict] | None = None    # [{label, bbox: [x1,y1,x2,y2] px}]
    class_names: list[str] | None = None
    image_size: list[int] | None = None
    qa_format: str | None = None          # ITT: multiple_choice | yes_no | numeric | open_ended
    is_nota: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _unit(*parts) -> float:
    return _rng(*parts).random()


def base_dir(data_dir: Path) -> Path:
    d = Path(data_dir)
    return d / "base-data" if (d / "base-data").is_dir() else d


def dataset_dirs(data_dir: Path, task: str) -> list[Path]:
    root = base_dir(data_dir) / task
    return sorted(p for p in root.iterdir() if (p / "samples.jsonl").exists() and (p / "meta.json").exists()) if root.is_dir() else []


def _read(ds_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((ds_dir / "meta.json").read_text())
    with open(ds_dir / "samples.jsonl") as f:
        return meta, [json.loads(l) for l in f]


def _classes(meta: dict, samples: list[dict]) -> list[str]:
    """Class labels as they appear in the data (sampling covers every class); catalog list as fallback."""
    labels = [s.get("label") for s in samples if s.get("label") is not None]
    seen, out = set(), []
    for c in labels or agtask.split_classes(meta["catalog"].get("classes")):
        if c is not None and norm(c) and norm(c) not in seen:
            seen.add(norm(c)); out.append(str(c))
    return out


def _cut(samples: list[dict], seed: int, limit: int | None) -> list[dict]:
    if not limit:
        return samples
    return sorted(samples, key=lambda s: _unit(seed, s["sample_id"]))[:limit]


class AcrossPool:
    """Class labels from every classification dataset in the data dir, grouped by ag-task bucket.
    Built from the whole data dir so options do not depend on --only filters."""

    def __init__(self, data_dir: Path):
        self.by_bucket: dict[str, list[tuple[str, str]]] = {}
        for d in dataset_dirs(data_dir, "classification"):
            meta, samples = _read(d)
            for c in _classes(meta, samples):
                b = agtask.class_bucket(meta["name"], meta["catalog"], c)
                self.by_bucket.setdefault(b, []).append((meta["name"], c))

    def draw(self, rng: random.Random, bucket: str, dataset: str, exclude: list[str], true_label: str | None, k: int) -> list[str]:
        excluded = {norm(e) for e in exclude}
        healthy = "healthy" in norm(true_label or "")

        def ok(c):
            n = norm(c)
            return (n not in excluded and not (healthy and "healthy" in n)
                    and (true_label is None or similarity(c, true_label) < 85))

        picked, seen = [], set()
        for pool in (self.by_bucket.get(bucket, []), [x for b, v in sorted(self.by_bucket.items()) if b != bucket for x in v]):
            order = list(range(len(pool)))
            rng.shuffle(order)
            for i in order:
                if len(picked) == k:
                    return picked
                ds, c = pool[i]
                if ds != dataset and norm(pretty(c)) not in seen and ok(c):
                    seen.add(norm(pretty(c))); picked.append(c)
        return picked


@dataclass
class MCQAOptions:
    """Ablations for every multiple-choice item (classification MCQA and ITT multiple choice). Defaults reproduce
    the standard benchmark.
    nota_rate: share of items whose true option is removed, so NOTA is correct (nested: items NOTA at a lower rate
        stay NOTA at a higher one). None = the defaults: 0.3 for classification MCQA, 0 for ITT (single-answer only).
    order_seed: reshuffle only the option order (same options, same NOTA items); None keeps the standard order.
    nota_position: "end" (NOTA, IDK last) or "random" (placed among the options).
    idk: offer "I don't know"."""
    nota_rate: float | None = None
    order_seed: int | None = None
    nota_position: str = "end"
    idk: bool = True


def _mcqa(s, meta, classes, fmt, seed, pool, opt: MCQAOptions = MCQAOptions()) -> EvalItem:
    p = prompts()
    label = s["label"]
    bucket = agtask.class_bucket(meta["name"], meta["catalog"], label)
    crop, src = crops.resolve(meta["catalog"], bucket, [label])
    rng = _rng(seed, s["sample_id"], fmt)
    others = [c for c in classes if norm(c) != norm(label)]
    nota = bool(others) and _unit(seed, s["sample_id"], fmt, "nota") < (NOTA_SHARE if opt.nota_rate is None else opt.nota_rate)
    k = N_OPTIONS if nota else N_OPTIONS - 1
    if fmt == "mcqa_within":
        distractors = rng.sample(others, min(k, len(others)))
    else:
        distractors = pool.draw(rng, bucket, meta["name"], classes, label, k)
    opts = [pretty(c) for c in distractors] + ([] if nota else [pretty(label)])
    rng.shuffle(opts)
    tail = [p["nota"]] + ([p["idk"]] if opt.idk else [])
    if opt.order_seed is None and opt.nota_position == "end":
        choices = opts + tail
    else:
        orng = _rng("order", seed if opt.order_seed is None else opt.order_seed, s["sample_id"], fmt)
        orng.shuffle(opts)
        choices = opts + tail
        if opt.nota_position == "random":
            orng.shuffle(choices)
    return EvalItem(
        item_id=f"{s['sample_id']}:{fmt}", sample_id=s["sample_id"], dataset=meta["name"], task="classification", format=fmt,
        images=s["images"], prompt=p["mcqa"].format(a_crop=crops.phrase(crop), options=", ".join(choices)),
        ag_task=bucket, crop=crop, crop_source=src, choices=choices, answer=p["nota"] if nota else pretty(label),
        gt_label=label, class_names=[pretty(c) for c in classes], image_size=s.get("image_size"), is_nota=nota)


def _oeq(s, meta, classes) -> EvalItem:
    label = s["label"]
    bucket = agtask.class_bucket(meta["name"], meta["catalog"], label)
    crop, src = crops.resolve(meta["catalog"], bucket, [label])
    return EvalItem(
        item_id=f"{s['sample_id']}:oeq", sample_id=s["sample_id"], dataset=meta["name"], task="classification", format="oeq",
        images=s["images"], prompt=prompts()["oeq"][bucket].format(a_crop=crops.phrase(crop)), ag_task=bucket, crop=crop,
        crop_source=src, answer=pretty(label), gt_label=label, class_names=[pretty(c) for c in classes], image_size=s.get("image_size"))


def _xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def _value_hit(m: np.ndarray, key: str) -> np.ndarray | None:
    """Pixels matching a mask key: 'v' (index), 'r,g,b' (color), or 'lo-hi' (range on the first channel)."""
    if re.fullmatch(r"\d+,\d+,\d+", key):
        rgb = m if m.ndim == 3 else np.stack([m] * 3, -1)
        return np.all(rgb[..., :3] == np.array([int(v) for v in key.split(",")]), axis=-1)
    m = m[..., 0] if m.ndim == 3 else m
    if re.fullmatch(r"\d+", key):
        return m == int(key)
    if r := re.fullmatch(r"(\d+)-(\d+)", key):
        return (m >= int(r[1])) & (m <= int(r[2]))
    return None


def mask_boxes(mask_path: Path, mask_classes: dict[str, str]) -> list[dict]:
    """One box per connected component per class. Keys mapped to the same name are merged first."""
    from scipy import ndimage

    m = np.array(Image.open(mask_path))
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    min_area = max(16, MIN_COMPONENT_FRAC * m.shape[0] * m.shape[1])
    by_name: dict[str, np.ndarray] = {}
    for key, name in mask_classes.items():
        if NON_CLASSES.match(str(name)) or (hit := _value_hit(m, str(key))) is None:
            continue
        by_name[name] = by_name[name] | hit if name in by_name else hit
    out = []
    for name, hit in by_name.items():
        lab, n = ndimage.label(hit, structure=np.ones((3, 3)))
        if not n:
            continue
        areas = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for i, sl in enumerate(ndimage.find_objects(lab)):
            if sl is not None and areas[i] >= min_area:
                out.append({"label": str(name), "bbox": [float(sl[1].start), float(sl[0].start), float(sl[1].stop), float(sl[0].stop)]})
    return out


def _grounding(s, meta, ds_dir, task) -> EvalItem | None:
    info = meta.get("info") or {}
    if task == "detection":
        names = info.get("class_names") or [str(c) for c in meta["catalog"].get("classes") or []]
        gt = [{"label": o["label"], "bbox": _xywh_to_xyxy(o["bbox"])} for o in s.get("objects", [])]
    else:
        named = seg_classes().get(meta["name"])
        mask_classes = {str(k): v for k, v in named.items()} if named else (info.get("mask_classes") or {})
        names = [n for n in mask_classes.values() if not NON_CLASSES.match(str(n))]
        if not names or not s.get("mask"):
            return None
        gt = mask_boxes(ds_dir / s["mask"], mask_classes)
    names = list(dict.fromkeys(map(str, names)))
    labels = [g["label"] for g in gt]
    bucket = agtask.objects_bucket(meta["name"], meta["catalog"], labels)
    crop, src = crops.resolve(meta["catalog"], bucket, labels or names)
    shown = ", ".join(pretty(n) for n in names)
    return EvalItem(
        item_id=f"{s['sample_id']}:grounding", sample_id=s["sample_id"], dataset=meta["name"], task=task, format="grounding",
        images=s["images"], prompt=prompts()["grounding"].format(class_names=shown, a_crop=crops.phrase(crop), box_order=prompts()["box_orders"]["yxyx"]), ag_task=bucket,
        crop=crop, crop_source=src, gt_boxes=gt, class_names=names, image_size=s.get("image_size"))


OPTION_RE = re.compile(r"^\s*([A-P])[\.\)]\s*(.+?)\s*$", re.M)


def _itt_reshape(q: str, opts: list[tuple[str, str]], ans: list[str], sid: str, seed: int, opt: MCQAOptions):
    """Rebuild an ITT multiple-choice question under the ablation options; returns (question, choices, answer letters)."""
    p, L = prompts(), "ABCDEFGHIJKLMNOP"
    stem = OPTION_RE.sub("", q).rstrip()
    texts = [t for _, t in opts]
    gold = [texts[L.index(a)] for a in ans]
    native_nota = next((t for t in texts if re.match(r"none of (the )?above", norm(t))), None)
    nota_text = native_nota or p["nota"]
    rate = 0.0 if opt.nota_rate is None else opt.nota_rate
    if len(gold) == 1 and not native_nota and len(texts) > 2 and _unit(seed, sid, "itt", "nota") < rate:
        texts, gold = [t for t in texts if t != gold[0]], [nota_text]
    body = [t for t in texts if t != native_nota]
    tail = [nota_text] + ([p["idk"]] if opt.idk else [])
    if opt.order_seed is not None or opt.nota_position == "random":
        orng = _rng("order", seed if opt.order_seed is None else opt.order_seed, sid, "itt")
        orng.shuffle(body)
        choices = body + tail
        if opt.nota_position == "random":
            orng.shuffle(choices)
    else:
        choices = body + tail
    lines = "\n".join(f"{L[i]}. {t}" for i, t in enumerate(choices))
    return f"{stem}\n{lines}", choices, sorted(L[choices.index(g)] for g in gold)


def _itt(s, meta, seed: int = 0, opt: MCQAOptions = MCQAOptions()) -> EvalItem:
    p = prompts()
    q = re.sub(r"<image>", "", s["question"])
    q = re.sub(r"image\d+:\s*,?\s*", "", q).strip()
    answer, fmt = str(s.get("answer") or ""), (s.get("strata") or {}).get("qa_format", "open_ended")
    choices, gt, is_nota = None, answer, False
    if fmt == "multiple_choice":
        opts = OPTION_RE.findall(q)
        letters = [l for l, _ in opts]
        ans = sorted(set(re.findall(r"^\s*([A-P])[\.\)]", answer, re.M)) or ([answer.strip()] if answer.strip() in letters else []))
        if len(opts) >= 2 and ans and set(ans) <= set(letters) and letters == list("ABCDEFGHIJKLMNOP"[:len(letters)]) \
                and opt != MCQAOptions():
            q, choices, gt = _itt_reshape(q, opts, ans, s["sample_id"], seed, opt)
            is_nota = [choices["ABCDEFGHIJKLMNOP".index(g)] for g in gt] == [p["nota"]] or any(
                re.match(r"none of (the )?above", norm(choices["ABCDEFGHIJKLMNOP".index(g)])) for g in gt)
        elif len(opts) >= 2 and ans and set(ans) <= set(letters) and letters == list("ABCDEFGHIJKLMNOP"[:len(letters)]):
            choices = [t for _, t in opts]
            extra = [x for x in (p["nota"], p["idk"]) if not (x == p["nota"] and any(re.match(r"none of (the )?above", norm(t)) for t in choices))]
            for x in extra:
                q += f"\n{'ABCDEFGHIJKLMNOP'[len(choices)]}. {x}"
                choices.append(x)
            gt = ans
        else:
            fmt = "open_ended"
    elif fmt == "yes_no":
        gt = parse_yes_no(answer)
        fmt = fmt if gt else "open_ended"
    elif fmt == "numeric":
        gt = parse_number(answer)
        fmt = fmt if gt is not None else "open_ended"
    if fmt == "open_ended":
        gt = answer
    suffix = p["itt_suffix"].get(fmt) or ""
    bucket = agtask.itt_bucket(s)
    crop, src = crops.resolve(meta["catalog"], bucket, sample=s)
    return EvalItem(
        item_id=f"{s['sample_id']}:native", sample_id=s["sample_id"], dataset=meta["name"], task="image_text_to_text",
        format="native", images=s["images"], prompt=(q + ("\n\n" + suffix if suffix else "")).strip(), ag_task=bucket,
        crop=crop, crop_source=src, choices=choices, answer=gt, qa_format=fmt, is_nota=is_nota,
        meta={"question": s["question"], "reference": answer})


def build_items(data_dir: Path, tasks=None, formats=None, only: str | None = None, limit_per_dataset: int | None = None,
                seed: int = 0, mcqa: MCQAOptions | None = None) -> tuple[list[EvalItem], list[dict]]:
    """Items for the selected tasks/formats, plus a list of skipped datasets with reasons."""
    tasks = tasks or TASKS
    items, skipped, pool = [], [], None
    for task in tasks:
        fmts = [f for f in FORMATS[task] if not formats or f in formats]
        if not fmts:
            continue
        for d in dataset_dirs(data_dir, task):
            if only and not re.search(only, d.name):
                continue
            meta, all_samples = _read(d)
            samples = _cut(all_samples, seed, limit_per_dataset)
            n0 = len(items)
            if task == "classification":
                classes = _classes(meta, all_samples)
                if "mcqa_across" in fmts and pool is None:
                    pool = AcrossPool(data_dir)
                for s in samples:
                    for f in fmts:
                        items.append(_oeq(s, meta, classes) if f == "oeq" else _mcqa(s, meta, classes, f, seed, pool, mcqa or MCQAOptions()))
            elif task == "image_text_to_text":
                its = [_itt(s, meta, seed, mcqa or MCQAOptions()) for s in samples]
                items += its if "native" in fmts else [it for it in its if it.qa_format == "multiple_choice"]
            else:
                items += [it for s in samples if (it := _grounding(s, meta, d, task)) is not None]
            for it in items[n0:]:
                it.images = [str(d / im) for im in it.images]
            if len(items) == n0:
                skipped.append({"dataset": d.name, "task": task, "reason": "no usable classes" if task == "segmentation" else "no samples"})
    return items, skipped


def spread(items: list[EvalItem], n: int, seed: int = 0) -> list[EvalItem]:
    """Up to n items, round-robin across datasets (for profiling)."""
    by_ds: dict[str, list[EvalItem]] = {}
    for it in sorted(items, key=lambda it: _unit(seed, it.item_id)):
        by_ds.setdefault(it.dataset, []).append(it)
    out, queues = [], [by_ds[k] for k in sorted(by_ds, key=lambda k: _unit(seed, k))]
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out
