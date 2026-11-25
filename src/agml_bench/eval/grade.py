"""Score one prediction against its item. Open-ended answers get a fuzzy score here and a judge verdict later."""

from PIL import Image

from agml_bench.eval import scoring as sc
from agml_bench.eval.client import Prediction
from agml_bench.eval.items import EvalItem
from agml_bench.eval.resources import prompts

INVALID = "__invalid__"


def _base(item: EvalItem, pred: Prediction, answer: str) -> dict:
    return {
        "item_id": item.item_id, "sample_id": item.sample_id, "dataset": item.dataset, "task": item.task,
        "format": item.format, "ag_task": item.ag_task, "crop": item.crop, "crop_source": item.crop_source,
        "qa_format": item.qa_format, "is_nota": item.is_nota, "gt": item.answer if item.task != "detection" else None,
        "answer": answer, "text": pred.text, "reasoning": pred.reasoning, "finish_reason": pred.finish_reason,
        "truncated": pred.finish_reason == "length", "error": pred.error, "prompt_tokens": pred.prompt_tokens,
        "completion_tokens": pred.completion_tokens, "latency_s": round(pred.latency_s, 3),
        "needs_judge": False, "judge_correct": None,
    }


def _scale_boxes(boxes: list[dict], sent: list[int] | None, orig: list[int] | None) -> list[dict]:
    if not sent or not orig or sent == orig:
        return boxes
    fx, fy = orig[0] / sent[0], orig[1] / sent[1]
    return [{**b, "bbox": [b["bbox"][0] * fx, b["bbox"][1] * fy, b["bbox"][2] * fx, b["bbox"][3] * fy]} for b in boxes]


def grade(item: EvalItem, pred: Prediction | str, key_formats: dict | None = None) -> dict:
    pred = pred if isinstance(pred, Prediction) else Prediction(text=str(pred))
    ans = sc.final_answer(pred.text)
    r = _base(item, pred, ans)
    p = prompts()

    if item.task == "classification" and item.format != "oeq":
        matched = sc.match_choice(ans, item.choices)
        r.update(pred_label=matched or INVALID, gt_label=item.answer, correct=matched == item.answer,
                 parse_ok=matched is not None, idk=matched == p["idk"])
    elif item.task == "classification":
        closest = sc.match_choice(ans, item.class_names) if ans else None
        r.update(pred_class=closest, gt_label=item.answer, class_names=item.class_names, correct=bool(ans) and sc.fuzzy_correct(ans, item.answer),
                 parse_ok=bool(ans), needs_judge=bool(ans), idk="know" in ans.lower() and "don" in ans.lower())
    elif item.task in ("detection", "segmentation"):
        orig = item.image_size or list(Image.open(item.images[0]).size)
        sent = pred.sent_size or orig
        boxes = sc.parse_boxes(pred.text, sent[0], sent[1], key_formats)
        r["parse_ok"] = boxes is not None
        boxes = _scale_boxes(boxes or [], sent, orig)
        m = sc.match_boxes(boxes, item.gt_boxes, item.class_names)
        r.update(pred_boxes=[{**b, "bbox": [round(v, 1) for v in b["bbox"]]} for b in boxes], gt_boxes=item.gt_boxes,
                 correct=m["acc"], **m)
    else:
        f = item.qa_format
        if f == "multiple_choice":
            letters = sc.parse_letters(ans, len(item.choices))
            if not letters and (m := sc.match_choice(ans, item.choices)):
                letters = ["ABCDEFGHIJKLMNOP"[item.choices.index(m)]]
            r.update(pred_label=",".join(letters) or INVALID, gt_label=",".join(item.answer),
                     correct=set(letters) == set(item.answer), parse_ok=bool(letters),
                     idk=bool(letters) and item.choices["ABCDEFGHIJKLMNOP".index(letters[0])] == p["idk"])
        elif f == "yes_no":
            yn = sc.parse_yes_no(ans)
            r.update(pred_label=yn or INVALID, gt_label=item.answer, correct=yn == item.answer, parse_ok=yn is not None)
        elif f == "numeric":
            num = sc.parse_number(ans)
            r.update(pred_label=num, gt_label=item.answer, correct=sc.numeric_correct(num, item.answer), parse_ok=num is not None)
        else:
            r.update(correct=bool(ans) and sc.fuzzy_correct(ans, item.answer), token_f1=round(sc.token_f1(ans, item.answer), 4),
                 rouge_l=round(sc.rouge_l(ans, item.answer), 4),
                     parse_ok=bool(ans), needs_judge=bool(ans), question=item.meta.get("question"))
    r["correct"] = float(r["correct"]) if item.task in ("detection", "segmentation") else bool(r["correct"])
    return r
