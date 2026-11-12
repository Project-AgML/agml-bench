"""Few-shot supervised baselines: one YOLO11 model per dataset, trained on k images per class from train-data/
(agbench sample-train) and scored on the evaluation items by the same grader as the VLMs.

Classification  YOLO11-cls. MCQA: the listed class with the highest probability (it never answers "None of the
                above" or "I don't know", so withheld-class items are always wrong). OEQ: the most probable class of the
                dataset, scored by exact match.
Detection       YOLO11 on the ground-truth boxes.
Segmentation    YOLO11-seg on polygons of the mask components (the same components the ground-truth boxes come from);
                predictions are scored through their boxes, like the VLMs.

usage: agbench eval yolo --data-dir $DATA --shots 5 --subset 0 [--task ...] [--only ...]
"""

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from agbench.eval.items import NON_CLASSES, _value_hit, build_items, mask_boxes
from agbench.eval.methods import Method
from agbench.eval.resources import norm, pretty, prompts, seg_classes
from agbench.progress import log

TASK_YOLO = {"classification": "-cls", "detection": "", "segmentation": "-seg"}
IMGSZ = {"classification": 224, "detection": 640, "segmentation": 640}


def train_items(train_root: Path, task: str, only: str | None, ids: set[str]):
    fmts = ["oeq"] if task == "classification" else ["grounding"]
    items, _ = build_items(train_root, [task], fmts, only)
    return [it for it in items if it.sample_id in ids]


def subset_ids(train_root: Path, task: str, dataset: str, subset: int, shots: int) -> set[str]:
    p = Path(train_root) / task / dataset / "subsets.json"
    return set(json.loads(p.read_text())[f"s{subset}"][str(shots)]) if p.exists() else set()


def _link(src: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(os.path.abspath(src), dst)


def _seg_polygons(mask_path: str, dataset: str, info: dict, names: list[str]) -> list[tuple[int, np.ndarray]]:
    """(class index, normalized polygon) per mask component, from the same class mapping as the evaluation boxes."""
    import cv2

    named = seg_classes().get(dataset)
    mask_classes = {str(k): v for k, v in named.items()} if named else (info.get("mask_classes") or {})
    m = np.array(Image.open(mask_path))
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    h, w = m.shape[:2]
    by_name = {}
    for key, name in mask_classes.items():
        if NON_CLASSES.match(str(name)) or (hit := _value_hit(m, str(key))) is None:
            continue
        by_name[str(name)] = by_name[str(name)] | hit if str(name) in by_name else hit
    out = []
    min_area = max(16, 0.0002 * h * w)
    for name, hit in by_name.items():
        if name not in names:
            continue
        contours, _ = cv2.findContours(hit.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= min_area and len(c) >= 3:
                out.append((names.index(name), (c.reshape(-1, 2) / [w, h]).clip(0, 1)))
    return out


def prepare(task: str, items, names: list[str], root: Path, dataset_dir: Path | None = None) -> str:
    """Write an ultralytics dataset under root; returns the path to train on."""
    shutil.rmtree(root, ignore_errors=True)
    if task == "classification":
        index = {norm(n): i for i, n in enumerate(names)}  # names: the evaluation items' (pretty) class names
        for split in ("train", "val"):  # no held-out data in few-shot training: val mirrors train, used only for logging
            for it in items:
                if (c := index.get(norm(pretty(it.gt_label)))) is not None:
                    _link(it.images[0], root / split / f"c{c}" / (it.sample_id + Path(it.images[0]).suffix))
        return str(root)
    meta = json.loads((dataset_dir / "meta.json").read_text())
    samples = {json.loads(l)["sample_id"]: json.loads(l) for l in open(dataset_dir / "samples.jsonl")}
    for it in items:
        img = root / "images" / (it.sample_id + Path(it.images[0]).suffix)
        _link(it.images[0], img)
        w, h = it.image_size or Image.open(it.images[0]).size
        lines = []
        if task == "detection":
            for b in it.gt_boxes:
                if b["label"] in names:
                    x1, y1, x2, y2 = b["bbox"]
                    lines.append(f"{names.index(b['label'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        else:
            mask = dataset_dir / samples[it.sample_id]["mask"]
            for ci, poly in _seg_polygons(str(mask), it.dataset, meta.get("info") or {}, names):
                lines.append(f"{ci} " + " ".join(f"{v:.6f}" for v in poly.flatten()))
        lab = root / "labels" / (it.sample_id + ".txt")
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml = root / "data.yaml"
    yaml.write_text(json.dumps({"path": str(root), "train": "images", "val": "images",
                                "names": {i: n for i, n in enumerate(names)}}))
    return str(yaml)


def device() -> int | str:
    """GPU 0, or the CPU only when AGBENCH_YOLO_ALLOW_CPU=1. Ultralytics silently falls back to the CPU when CUDA
    fails to initialise (e.g. a driver/toolkit mismatch on an older GPU), which would make a run look merely slow."""
    import torch

    if torch.cuda.is_available():
        return 0
    if os.environ.get("AGBENCH_YOLO_ALLOW_CPU") == "1":
        return "cpu"
    raise RuntimeError("CUDA is not available to PyTorch (torch %s, built for CUDA %s); refusing to train on the CPU. "
                       "Set AGBENCH_YOLO_ALLOW_CPU=1 to allow it." % (torch.__version__, torch.version.cuda))


def train(task: str, data: str, out: Path, size: str, epochs: int, weights_dir: Path, seed: int) -> Path:
    from ultralytics import YOLO

    base = weights_dir / f"yolo11{size}{TASK_YOLO[task]}.pt"
    base.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base) if base.exists() else base.name)
    if not base.exists() and Path(base.name).exists():  # ultralytics downloads into the working directory
        shutil.move(base.name, base)
        model = YOLO(str(base))
    model.train(data=data, epochs=epochs, imgsz=IMGSZ[task], project=str(out.parent), name=out.name, exist_ok=True, device=device(),
                seed=seed, deterministic=True, val=False, plots=False, verbose=False,
                workers=int(os.environ.get("AGBENCH_YOLO_WORKERS", "2")),  # data-loading processes per training; few images, so 2 suffice
                cache=os.environ.get("AGBENCH_YOLO_CACHE") or False,  # "ram": decode the few training images once, not every epoch
                batch=16 if task == "classification" else 8)
    return Path(model.trainer.save_dir) / "weights" / "last.pt"


class YOLOMethod(Method):
    """Predictions from per-dataset trained weights ({weights}/{dataset}/weights/last.pt)."""

    def __init__(self, weights: dict[str, Path], names: dict[str, list[str]], label: str, subset: int):
        self.weights, self.names, self.subset = weights, names, subset
        self.name = label
        self._cache = (None, None)

    def config(self) -> dict:
        return {"name": self.name, "baseline": "yolo", "gen_seed": self.subset}

    def _model(self, dataset: str):
        from ultralytics import YOLO

        if self._cache[0] != dataset:
            self._cache = (dataset, YOLO(str(self.weights[dataset])))
        return self._cache[1]

    def predict(self, item) -> str:
        m, names = self._model(item.dataset), self.names[item.dataset]
        if item.task == "classification":
            raw = m.predict(item.images[0], imgsz=IMGSZ["classification"], device=device(), verbose=False)[0].probs.data.cpu().numpy()
            # model outputs follow its own (sorted) folder names "c<index>"; classes without training images are absent
            probs = np.full(len(names), -1.0)
            for j, folder in m.names.items():
                probs[int(folder[1:])] = raw[j]
            if item.format == "oeq":
                return names[int(probs.argmax())]
            p = prompts()
            index = {norm(n): i for i, n in enumerate(names)}
            listed = [c for c in item.choices if c not in (p["nota"], p["idk"]) and norm(c) in index]
            return max(listed, key=lambda c: probs[index[norm(c)]]) if listed else p["idk"]
        r = m.predict(item.images[0], imgsz=IMGSZ[item.task], device=device(), verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        cls = r.boxes.cls.cpu().numpy().astype(int).tolist() if r.boxes is not None else []
        return json.dumps([{"bbox": [round(v, 1) for v in b], "label": names[c]} for b, c in zip(boxes, cls)])


def dataset_names(item_list) -> dict[str, list[str]]:
    """Class list per dataset, in the order used for training (the evaluation items' class names)."""
    out = {}
    for it in item_list:
        if it.dataset in out:
            continue
        if it.task == "classification" and it.format != "oeq":
            continue
        out[it.dataset] = list(it.class_names)
    return out
