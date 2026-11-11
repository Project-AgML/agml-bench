"""Naive baselines: answers that use no image, scored by the same grader as the models.

chance  A uniformly random answer: one of the listed options (MCQA, VQA multiple choice; "None of the above" and
        "I don't know" included), one of the dataset's classes (OEQ), yes or no, one of the dataset's numeric answers,
        and one whole-image box with a random class (detection, segmentation). Seeded; replicate with --seed.
prior   The best constant answer: always "None of the above" (MCQA), the dataset's most frequent class (OEQ), one
        whole-image box with the most frequent class (detection, segmentation), and the most frequent letter, yes/no
        answer or the median number among the dataset's questions (VQA). The VQA priors are taken from the evaluated
        questions themselves, so they bound what a label prior could reach.
Open-ended VQA gets no answer (scored wrong): a constant text has no meaningful verdict.
OEQ answers are exact class names, so they are scored by exact match instead of the LLM judge.
"""

import json
import random
import statistics
from collections import Counter, defaultdict

from agbench.eval.items import EvalItem
from agbench.eval.methods import Method, item_seed
from agbench.eval.resources import norm, prompts

LETTERS = "ABCDEFGHIJKLMNOP"
KINDS = ("chance", "prior")


def priors(items: list[EvalItem]) -> dict:
    """Per dataset: most frequent OEQ class / box class / VQA letter and yes-no answer, and the numeric answers."""
    c = defaultdict(Counter)
    nums = defaultdict(list)
    for it in items:
        d = it.dataset
        if it.task == "classification" and it.format == "oeq":
            c[(d, "oeq")][it.answer] += 1
        elif it.task in ("detection", "segmentation"):
            c[(d, "box")].update(b["label"] for b in it.gt_boxes or [])
        elif it.qa_format == "multiple_choice":
            c[(d, "mc")][",".join(it.answer)] += 1
        elif it.qa_format == "yes_no":
            c[(d, "yn")][it.answer] += 1
        elif it.qa_format == "numeric":
            nums[d].append(float(it.answer))
    # most_common breaks ties by first occurrence; sort keys first so the choice does not depend on item order
    top = {k: max(sorted(v), key=lambda x: v[x]) for k, v in c.items()}
    return {"top": top, "nums": dict(nums)}


def _box(item: EvalItem, label: str) -> str:
    return json.dumps([{"box_2d": [0, 0, 1000, 1000], "label": label}])  # whole image, yxyx on 0-1000


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


class NaiveBaseline(Method):
    def __init__(self, kind: str, stats: dict, seed: int = 0):
        assert kind in KINDS, kind
        self.kind, self.stats, self.seed = kind, stats, seed
        self.name = f"baseline/{kind}"

    def config(self) -> dict:
        return {"name": self.name, "baseline": self.kind, "gen_seed": self.seed if self.kind == "chance" else None}

    def predict(self, item: EvalItem) -> str:
        rng = random.Random(item_seed(self.seed, item.item_id))
        top, d = self.stats["top"], item.dataset
        if item.task == "classification" and item.format != "oeq":
            return rng.choice(item.choices) if self.kind == "chance" else prompts()["nota"]
        if item.task == "classification":
            return rng.choice(item.class_names) if self.kind == "chance" else top[(d, "oeq")]
        if item.task in ("detection", "segmentation"):
            labels = item.class_names or ["object"]
            return _box(item, rng.choice(labels) if self.kind == "chance" else top.get((d, "box"), labels[0]))
        f = item.qa_format
        if f == "multiple_choice":
            return LETTERS[rng.randrange(len(item.choices))] if self.kind == "chance" else top[(d, "mc")]
        if f == "yes_no":
            return rng.choice(["yes", "no"]) if self.kind == "chance" else top[(d, "yn")]
        if f == "numeric":
            nums = self.stats["nums"][d]
            return _num(rng.choice(nums) if self.kind == "chance" else statistics.median(nums))
        return ""


def exact_oeq_verdicts(records: list[dict]) -> list[dict]:
    """OEQ answers of a baseline are class names: correct iff the name is the reference class."""
    for r in records:
        if r["task"] == "classification" and r["format"] == "oeq":
            r["needs_judge"] = True  # graded by this verdict, not by the fuzzy score
            r["judge_correct"] = bool(r["answer"]) and norm(r["answer"]) == norm(r["gt"])
            r["judge_model"] = "exact match (baseline)"
    return records
