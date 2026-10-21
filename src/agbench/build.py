"""Index a dataset's labels (no image bytes), then extract only the selected samples."""

import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from agbench.hub import IMAGE_EXTS, Repo, classlabel_names, feature_type, image_format
from agbench.progress import Progress
from agbench.sampling import qa_format

MAX_MASK_VALUES = 64


class Unsupported(Exception):
    pass


def sample_id(name: str, ref: dict) -> str:
    key = f"{ref['file']}:{ref.get('row', ref.get('path', ''))}"
    return f"{name}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def _name_for(idx, names: list[str] | None, fallback: list[str]) -> str:
    if isinstance(idx, str) and not idx.lstrip("-").isdigit():
        return idx
    i = int(float(idx))
    for src in (names, fallback):
        if src and 0 <= i < len(src):
            return str(src[i])
    return str(i)


def _image_col(feats: dict) -> str | None:
    if "image" in feats:
        return "image"
    col = next((k for k, v in feats.items() if feature_type(v) == "Image" and k != "mask"), None)
    if col is None and {"bands", "bands_shape", "bands_dtype"} <= feats.keys():
        return "bands"
    return col


BAND_COLS = ["bands", "bands_shape", "bands_dtype", "band_metadata"]


def bands_to_png(row: dict) -> bytes | None:
    """Raw GeoTIFF band array (bands, H, W) -> RGB PNG, using colorinterp when present."""
    if not row.get("bands"):
        return None
    arr = np.frombuffer(row["bands"], dtype=row["bands_dtype"]).reshape(row["bands_shape"])
    interp = (json.loads(row.get("band_metadata") or "{}") or {}).get("colorinterp") or []
    rgb = [interp.index(c) for c in ("red", "green", "blue") if c in interp]
    idx = rgb if len(rgb) == 3 else [0, 1, 2] if arr.shape[0] >= 3 else [0, 0, 0]
    img = arr[idx].transpose(1, 2, 0)
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, (1, 99))
        img = (np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- parquet index

def index_parquet(repo: Repo, files, task: str, catalog_classes: list[str], seg_max_rows: int, rng) -> tuple[list[dict], dict]:
    feats = repo.hf_features(files[0].path)
    img_col = _image_col(feats)
    if img_col is None:
        raise Unsupported("no image column")
    info = {"layout": "parquet", "image_col": img_col}
    if task == "image_classification":
        col = "label" if "label" in feats else next((k for k, v in feats.items() if feature_type(v) == "ClassLabel"), None)
        if col is None:
            raise Unsupported("no label column")
        names = classlabel_names(feats[col])
        info.update(label_col=col, class_names=names or catalog_classes)
        return _scan(repo, files, [col], lambda row: _cls_record(row[col], names, catalog_classes)), info
    if task == "object_detection":
        if "objects" not in feats:
            raise Unsupported("no objects column")
        names = classlabel_names(feats["objects"])
        info.update(class_names=names or catalog_classes)
        return _scan(repo, files, ["objects"], lambda row: _det_record(row["objects"], names, catalog_classes)), info
    if task == "semantic_segmentation":
        if feature_type(feats.get("mask")) not in ("Image", None) or "mask" not in feats:
            raise Unsupported("no mask image column")
        info.update(mask_col="mask")
        recs = _scan_masks(repo, files, seg_max_rows, rng)
        info["mask_classes"] = mask_class_map(recs, catalog_classes)
        _apply_mask_names(recs, info["mask_classes"])
        return recs, info
    raise Unsupported(f"task {task}")


def _scan(repo: Repo, files, cols: list[str], make) -> list[dict]:
    recs = []
    prog = Progress("indexing files", len(files))
    for fr in files:
        pf = repo.parquet(fr.path)
        extra = ["split"] if "split" in pf.schema_arrow.names else []
        t = pf.read(columns=cols + extra).to_pylist()
        for row_i, row in enumerate(t):
            r = make(row)
            if r is None:
                continue
            r["ref"] = {"file": fr.path, "row": row_i, "config": fr.config, "split": row.get("split") or fr.split}
            recs.append(r)
        prog.add()
    return recs


def _cls_record(label, names, fallback) -> dict | None:
    if label is None:
        return None
    name = _name_for(label, names, fallback)
    return {"keys": [name], "ann": {"label": name}}


def _det_record(objects, names, fallback) -> dict | None:
    objects = objects or {}
    bboxes = objects.get("bbox") or []
    cats = objects.get("categories")
    if not isinstance(cats, list):
        cats = [cats] * len(bboxes)
    cat_names = objects.get("category_names")
    labels = [cat_names[i] if cat_names and i < len(cat_names) and cat_names[i] else _name_for(c, names, fallback)
              for i, c in enumerate(cats)]
    objs = [{"label": l, "bbox_raw": [float(v) for v in b]} for l, b in zip(labels, bboxes) if b is not None]
    return {"keys": sorted(set(o["label"] for o in objs)) or ["__empty__"], "ann": {"objects": objs}}


def _mask_keys(data: bytes) -> list[str]:
    with Image.open(io.BytesIO(data)) as im:
        if im.mode in ("RGB", "RGBA"):
            a = np.asarray(im.convert("RGB")).reshape(-1, 3)
            vals = np.unique(a[:: max(1, len(a) // 200000)], axis=0)
            keys = [",".join(map(str, v)) for v in vals]
        else:
            a = np.asarray(im if im.mode in ("L", "P", "I", "I;16") else im.convert("L"))
            keys = [str(v) for v in np.unique(a)]
    return keys if len(keys) <= MAX_MASK_VALUES else ["__many__"]


def _scan_masks(repo: Repo, files, max_rows: int, rng) -> list[dict]:
    groups = []
    for fr in files:
        md = repo.parquet(fr.path).metadata
        start = 0
        for g in range(md.num_row_groups):
            n = md.row_group(g).num_rows
            groups.append((fr, g, start, n))
            start += n
    order = rng.permutation(len(groups))
    recs, scanned = [], 0
    prog = Progress("scanning masks", min(max_rows, sum(g[3] for g in groups)))
    for gi in order:
        if scanned >= max_rows:
            break
        fr, g, start, n = groups[gi]
        pf = repo.parquet(fr.path)
        cols = ["mask"] + (["split"] if "split" in pf.schema_arrow.names else [])
        rows = pf.read_row_group(g, columns=cols).to_pylist()
        for j, row in enumerate(rows):
            m = row["mask"]
            if not m or not m.get("bytes"):
                continue
            recs.append({"keys": _mask_keys(m["bytes"]), "ann": {},
                         "ref": {"file": fr.path, "row": start + j, "config": fr.config, "split": row.get("split") or fr.split}})
        scanned += n
        prog.add(n)
    return recs


def mask_class_map(recs: list[dict], classes: list[str]) -> dict[str, str]:
    vals = sorted({k for r in recs for k in r["keys"]})
    if any("," in v for v in vals) or "__many__" in vals:
        return {v: v for v in vals}
    ints = sorted(int(v) for v in vals)
    lower = [c.lower() for c in classes]
    out = {}
    if classes and ("background" in lower or (ints and len(classes) == ints[-1] + 1)):
        out = {str(v): classes[v] if v < len(classes) else str(v) for v in ints}
    elif classes and ints and len(classes) == ints[-1]:
        out = {str(v): ("background" if v == 0 else classes[v - 1]) for v in ints}
    elif ints == [0, 255] and len(classes) == 1:
        out = {"0": "background", "255": classes[0]}
    else:
        out = {str(v): ("background" if v == 0 else f"value_{v}") for v in ints}
    return out


def _apply_mask_names(recs: list[dict], mapping: dict[str, str]) -> None:
    for r in recs:
        names = [mapping.get(k, k) for k in r["keys"]]
        fg = sorted(set(n for n in names if n.lower() != "background"))
        r["ann"]["classes_present"] = fg
        r["keys"] = fg or ["background"]


# ---------------------------------------------------------------- metadata.jsonl index

def index_metadata_jsonl(repo: Repo, task: str, catalog_classes: list[str], skip_keywords: list[str], seg_max_rows: int, rng):
    metas = [f for f in repo.files if f.endswith("metadata.jsonl") and not any(k in f.lower() for k in skip_keywords)]
    if not metas:
        raise Unsupported("no usable metadata.jsonl")
    recs = []
    for mpath in metas:
        base = mpath.rsplit("/", 1)[0] + "/" if "/" in mpath else ""
        for row_i, line in enumerate(repo.read_bytes(mpath).decode().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            ref = {"file": mpath, "row": row_i, "config": "default", "split": row.get("split", "train"),
                   "image_path": base + row["file_name"]}
            if task == "image_classification":
                lab = row.get("label", row.get("class"))
                r = _cls_record(lab, None, catalog_classes)
            elif task == "object_detection":
                r = _det_record(row.get("objects"), None, catalog_classes)
            elif task == "semantic_segmentation":
                mk = next((row[k] for k in ("mask_path", "mask_file_name", "mask", "annotation", "label_path")
                           if isinstance(row.get(k), str)), None)
                if not mk:
                    continue
                ref["mask_path"] = base + mk
                r = {"keys": [], "ann": {}}
            else:
                raise Unsupported(f"task {task}")
            if r is None:
                continue
            r["ref"] = ref
            recs.append(r)
    info = {"layout": "metadata_jsonl", "class_names": catalog_classes}
    if task == "semantic_segmentation":
        if len(recs) > seg_max_rows:
            recs = [recs[i] for i in sorted(rng.choice(len(recs), seg_max_rows, replace=False))]
        prog = Progress("scanning masks", len(recs))
        keys = []
        with ThreadPoolExecutor(16) as ex:
            for k in ex.map(lambda r: _mask_keys(repo.read_bytes(r["ref"]["mask_path"])), recs):
                keys.append(k)
                prog.add()
        for r, k in zip(recs, keys):
            r["keys"] = k
        info["mask_classes"] = mask_class_map(recs, catalog_classes)
        _apply_mask_names(recs, info["mask_classes"])
    return recs, info


# ---------------------------------------------------------------- image-text-to-text index

def _text_of(msg) -> str:
    return "\n".join(c.get("text") or "" for c in (msg.get("content") or []) if c.get("type") == "text")


def _qa(messages) -> tuple[str, str]:
    q = next((_text_of(m) for m in messages or [] if m.get("role") == "user"), "")
    a = next((_text_of(m) for m in messages or [] if m.get("role") == "assistant"), "")
    return q, a


def index_itt(repo: Repo, files, keep_keys: list[str]) -> tuple[list[dict], dict]:
    recs = []
    prog = Progress("indexing files", len(files))
    for fr in files:
        pf = repo.parquet(fr.path)
        names = pf.schema_arrow.names
        cols = [c for c in ("id", "messages", "raw_metadata", "origin_dataset") if c in names]
        for row_i, row in enumerate(pf.read(columns=cols).to_pylist()):
            try:
                meta = json.loads(row.get("raw_metadata") or "{}")
            except json.JSONDecodeError:
                meta = {}
            q, a = _qa(row.get("messages"))
            strata = {k: (str(meta[k]) if not isinstance(meta[k], (list, dict)) else None)
                      for k in keep_keys if k in meta and meta[k] not in (None, "")}
            strata = {k: v for k, v in strata.items() if v is not None and len(v) <= 80}
            if row.get("origin_dataset"):
                strata["origin_dataset"] = row["origin_dataset"]
            strata["config"] = fr.config
            strata["qa_format"] = qa_format(q, a)
            rid = row.get("id") or f"{fr.path}:{row_i}"
            strata["id_prefix"] = re.sub(r"[_\-]?\d+$", "", str(rid).rsplit("/", 1)[-1])
            recs.append({"keys": [], "strata": strata, "question": q[:500],
                         "ref": {"file": fr.path, "row": row_i, "config": fr.config, "split": fr.split}})
        prog.add()
    return recs, {"layout": "itt_parquet"}


# ---------------------------------------------------------------- extraction

def _rows_by_group(repo: Repo, path: str, rows: list[int]) -> dict[int, list[tuple[int, int]]]:
    md = repo.parquet(path).metadata
    bounds, start = [], 0
    for g in range(md.num_row_groups):
        n = md.row_group(g).num_rows
        bounds.append((start, start + n))
        start += n
    out = defaultdict(list)
    for r in rows:
        for g, (s, e) in enumerate(bounds):
            if s <= r < e:
                out[g].append((r, r - s))
                break
    return out


def fetch_parquet_rows(repo: Repo, path: str, rows: list[int], cols: list[str], prog: Progress | None = None) -> dict[int, dict]:
    pf = repo.parquet(path)
    cols = [c for c in cols if c in pf.schema_arrow.names]
    got = {}
    for g, pairs in _rows_by_group(repo, path, rows).items():
        t = pf.read_row_group(g, columns=cols)
        for r, local in pairs:
            got[r] = t.slice(local, 1).to_pylist()[0]
        if prog:
            prog.add(len(pairs))
    return got


def to_pixel_xywh(b: list[float], size, fmt: str | None) -> list[float]:
    w_img, h_img = size or (1, 1)
    normalized = all(0 <= v <= 1.0 for v in b)
    if fmt is None:
        fmt = "cxcywh" if normalized else "xywh"
    x, y, c, d = b
    if normalized:
        x, c = x * w_img, c * w_img
        y, d = y * h_img, d * h_img
    if fmt == "xyxy":
        return [x, y, c - x, d - y]
    if fmt == "cxcywh":
        return [x - c / 2, y - d / 2, c, d]
    return [x, y, c, d]


class Writer:
    def __init__(self, out_dir):
        self.out = out_dir
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

    def save(self, sub: str, stem: str, data: bytes) -> tuple[str, tuple | None]:
        ext, size = image_format(data)
        (self.out / sub).mkdir(exist_ok=True)
        rel = f"{sub}/{stem}.{ext}"
        (self.out / rel).write_bytes(data)
        return rel, size


def extract(repo: Repo, name: str, task: str, info: dict, recs: list[dict], out_dir, bbox_format: str | None) -> list[dict]:
    w = Writer(out_dir)
    samples = []
    by_file = defaultdict(list)
    for r in recs:
        by_file[r["ref"]["file"]].append(r)
    prog = Progress("fetching samples", len(recs))

    for path, group in by_file.items():
        if info["layout"] == "metadata_jsonl":
            def load(r):
                img = repo.read_bytes(r["ref"]["image_path"])
                mask = repo.read_bytes(r["ref"]["mask_path"]) if "mask_path" in r["ref"] else None
                return r, {"image": img, "mask": mask}
            loaded = []
            with ThreadPoolExecutor(8) as ex:
                for item in ex.map(load, group):
                    loaded.append(item)
                    prog.add()
        elif task == "image_text_to_text":
            rows = fetch_parquet_rows(repo, path, [r["ref"]["row"] for r in group],
                                      ["id", "images", "messages", "raw_metadata", "origin_dataset"], prog)
            loaded = [(r, rows[r["ref"]["row"]]) for r in group]
        else:
            bands = info["image_col"] == "bands"
            cols = (BAND_COLS if bands else [info["image_col"]]) + (["mask"] if task == "semantic_segmentation" else [])
            rows = fetch_parquet_rows(repo, path, [r["ref"]["row"] for r in group], cols, prog)
            loaded = [(r, {"image": bands_to_png(rows[r["ref"]["row"]]) if bands
                           else (rows[r["ref"]["row"]][info["image_col"]] or {}).get("bytes"),
                           "mask": (rows[r["ref"]["row"]].get("mask") or {}).get("bytes")})
                      for r in group]

        if task == "image_text_to_text":
            samples += _write_itt(repo, name, path, loaded, w)
            continue
        for r, row in loaded:
            sid = sample_id(name, r["ref"])
            if not row.get("image"):
                continue
            rel, size = w.save("images", sid, row["image"])
            s = {"sample_id": sid, "dataset": name, "task": task, "split": r["ref"]["split"],
                 "images": [rel], "image_size": list(size) if size else None, "ref": r["ref"],
                 "keys": r.get("keys", [])}
            ann = dict(r["ann"])
            if task == "object_detection":
                ann["objects"] = [{"label": o["label"], "bbox": to_pixel_xywh(o["bbox_raw"], size, bbox_format),
                                   "bbox_raw": o["bbox_raw"]} for o in ann["objects"]]
            if task == "semantic_segmentation" and row.get("mask"):
                ann["mask"], _ = w.save("masks", sid, row["mask"])
            s.update(ann)
            samples.append(s)
    return samples


def _read_shard_members(repo: Repo, wanted: dict[str, list[str]], workers: int = 8) -> dict[str, bytes]:
    """Read zip members over HTTP range requests; each thread keeps its own ZipFile handle."""
    import threading

    local = threading.local()

    def read(item):
        shard, member = item
        handles = local.__dict__.setdefault("zips", {})
        if shard not in handles:
            handles[shard] = zipfile.ZipFile(repo.open(shard, block_size=2**20))
        return member, handles[shard].read(member)

    items = [(sh, m) for sh, ms in wanted.items() for m in ms]
    prog = Progress("reading images from zip shards", len(items))
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for member, data in ex.map(read, items):
            out[member] = data
            prog.add()
    return out


def _write_itt(repo: Repo, name: str, path: str, loaded, w: Writer) -> list[dict]:
    base = path.rsplit("/", 1)[0] if "/" in path else ""
    shard_map: dict[str, str] = {}
    wanted: dict[str, list[str]] = defaultdict(list)
    loose: list[str] = []
    if any(im.get("bytes") is None and im.get("path") for _, row in loaded for im in row.get("images") or []):
        shard_map = repo.shard_index(base) or repo.shard_index("")
        for _, row in loaded:
            for im in row.get("images") or []:
                if im.get("bytes") is None and im.get("path"):
                    if im["path"] in shard_map:
                        wanted[shard_map[im["path"]]].append(im["path"])
                    elif im["path"] in repo.files:
                        loose.append(im["path"])
    fetched = _read_shard_members(repo, wanted) if wanted else {}
    if loose:
        with ThreadPoolExecutor(8) as ex:
            fetched.update(zip(loose, ex.map(repo.read_bytes, loose)))

    out = []
    for r, row in loaded:
        sid = sample_id(name, r["ref"])
        rels = []
        for k, im in enumerate(row.get("images") or []):
            data = im.get("bytes") or fetched.get(im.get("path"))
            if data:
                rels.append(w.save("images", f"{sid}_{k}", data)[0])
        try:
            meta = json.loads(row.get("raw_metadata") or "{}")
        except json.JSONDecodeError:
            meta = {"raw": row.get("raw_metadata")}
        q, a = _qa(row.get("messages"))
        out.append({"sample_id": sid, "dataset": name, "task": "image_text_to_text", "split": r["ref"]["split"],
                    "images": rels, "ref": r["ref"], "keys": [], "source_id": row.get("id"), "messages": row.get("messages"),
                    "question": q, "answer": a, "strata": r["strata"], "raw_metadata": meta})
    return out
