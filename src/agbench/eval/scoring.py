"""Answer extraction, choice matching, fuzzy match, and box scoring."""

import json
import re
from collections import Counter

from rapidfuzz import fuzz

from agbench.eval.resources import norm

FUZZY_THRESHOLD = 60  # paper: similarity >= 0.6 counts as a match
IOU_THRESHOLD = 0.5


def final_answer(text: str | None) -> str:
    """Strip reasoning; prefer the last <answer> tag."""
    text = text or ""
    tags = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if tags:
        return tags[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*</think>", "", text, flags=re.S | re.I)  # opening tag lost
    return text.replace("**", "").strip()


def similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b) if min(len(a), len(b)) >= 4 else 0)


def match_choice(pred: str, options: list[str]) -> str | None:
    """Exact -> substring (longest option mentioned) -> fuzzy >= threshold. Returns an option or None."""
    p = norm(pred)
    if not p:
        return None
    by_norm = {norm(o): o for o in options}
    if p in by_norm:
        return by_norm[p]
    mentioned = [o for n, o in by_norm.items() if n and re.search(rf"\b{re.escape(n)}\b", p)]
    if mentioned:
        return max(mentioned, key=lambda o: len(norm(o)))
    best = max(options, key=lambda o: similarity(pred, o))
    return best if similarity(pred, best) >= FUZZY_THRESHOLD else None


def fuzzy_correct(pred: str, gt: str) -> bool:
    return similarity(pred, gt) >= FUZZY_THRESHOLD or (norm(gt) and re.search(rf"\b{re.escape(norm(gt))}\b", norm(pred)) is not None)


def token_f1(pred: str, gt: str) -> float:
    p, g = norm(pred).split(), norm(gt).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, gt: str) -> float:
    """ROUGE-L F1: longest common token subsequence (no stemming)."""
    p, g = norm(pred).split(), norm(gt).split()
    if not p or not g:
        return 0.0
    prev = [0] * (len(g) + 1)
    for a in p:
        cur = [0]
        for j, b in enumerate(g):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    if not prev[-1]:
        return 0.0
    prec, rec = prev[-1] / len(p), prev[-1] / len(g)
    return 2 * prec * rec / (prec + rec)


# ---- letters / yes-no / numeric (ITT) ----

def parse_letters(pred: str, n_options: int) -> list[str]:
    L = "ABCDEFGHIJKLMNOP"[:n_options]
    head = pred.strip()
    m = re.match(rf"^\W*([{L}](?:\s*(?:,|and|&|/)?\s*[{L}])*)(?=\s*(?:$|[.):\]\n]))", head)
    if m:
        return sorted(set(re.findall(rf"[{L}]", m.group(1))))
    m = re.search(rf"(?:answer|option)s?\s*(?:is|are|:)?\s*\(?([{L}])(?![A-Za-z])", head, re.I)
    return [m.group(1)] if m else []


def parse_yes_no(pred: str) -> str | None:
    m = re.search(r"\b(yes|no)\b", pred.lower())
    return m.group(1) if m else None


def parse_number(pred: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    return float(m.group()) if m else None


def numeric_correct(pred: float | None, gt: float | None, rel_tol: float = 0.05) -> bool:
    if pred is None or gt is None:
        return False
    return abs(pred - gt) <= max(rel_tol * abs(gt), 0.5 if float(gt).is_integer() else 1e-6)


# ---- boxes ----

def _json_array(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r",\s*([\]}])", r"\1", text[start:end + 1]))
        except json.JSONDecodeError:
            return None


def parse_boxes(text: str, width: int, height: int, key_formats: dict | None = None) -> list[dict] | None:
    """Model JSON -> [{label, bbox: [x1,y1,x2,y2] px}] in the size the image was sent at. None if unparseable.

    box_2d: [y1,x1,y2,x2] on 0-1000 (prompted). bbox_2d / bbox: [x1,y1,x2,y2]; scale from key_formats
    (xyxy_1000 | xyxy_px), else values <= 1 are normalized, else pixels."""
    formats = {"box_2d": "yxyx_1000", **(key_formats or {})}
    arr = _json_array(final_answer(text))
    if arr is None:
        return [] if final_answer(text).strip() in ("[]", "") else None
    out = []
    for o in arr if isinstance(arr, list) else []:
        if not isinstance(o, dict):
            continue
        key = next((k for k in ("box_2d", "bbox_2d", "bbox", "box") if k in o), None)
        b = o.get(key) if key else None
        if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
            continue
        fmt = formats.get(key) or ("xyxy_norm" if max(b) <= 1 else "xyxy_px")
        if fmt == "yxyx_1000":
            y1, x1, y2, x2 = b
            box = [x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000]
        elif fmt == "xyxy_1000":
            box = [b[0] * width / 1000, b[1] * height / 1000, b[2] * width / 1000, b[3] * height / 1000]
        elif fmt == "xyxy_norm":
            box = [b[0] * width, b[1] * height, b[2] * width, b[3] * height]
        else:
            box = list(map(float, b))
        x1, y1, x2, y2 = min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])
        out.append({"label": str(o.get("label", "")), "bbox": [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]})
    return out


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred: list[dict], gt: list[dict], class_names: list[str]) -> dict:
    """Greedy one-to-one matching at IoU >= 0.5. Labels must agree when the dataset has more than one class."""
    multi = len(class_names) > 1
    plabel = [match_choice(p["label"], class_names) if multi else None for p in pred]
    pairs = sorted(((iou(p["bbox"], g["bbox"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)
                    if not multi or norm(plabel[i] or "") == norm(g["label"])), reverse=True)
    used_p, used_g, ious = set(), set(), []
    for s, i, j in pairs:
        if s < IOU_THRESHOLD:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); ious.append(s)
    tp = len(ious)
    fp, fn = len(pred) - tp, len(gt) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "mean_iou": sum(ious) / tp if tp else 0.0,
            "acc": tp / (tp + fp + fn) if tp + fp + fn else 1.0}
