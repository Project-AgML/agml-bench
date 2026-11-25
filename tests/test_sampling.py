from collections import Counter

from agml_bench.build import mask_class_map, to_pixel_xywh
from agml_bench.sampling import qa_format, rng_for, select_greedy_cover, select_itt, select_per_class

SPLITS = ["test", "validation", "train"]
ITT_CFG = {"per_stratum": 5, "max_per_dataset": 40, "primary_keys": ["task"], "secondary_keys": [],
           "max_stratum_values": 60, "fill_to_cap": False}


def rec(keys, split="train", **strata):
    return {"keys": keys, "ref": {"split": split}, "strata": strata, "question": ""}


def test_per_class_caps_and_is_deterministic():
    recs = [rec([c]) for c in "a" * 20 + "b" * 3 + "c" * 9]
    a = select_per_class(recs, 5, rng_for(0, "x"), SPLITS)
    b = select_per_class(recs, 5, rng_for(0, "x"), SPLITS)
    assert a == b
    assert Counter(recs[i]["keys"][0] for i in a) == {"a": 5, "b": 3, "c": 5}
    assert a != select_per_class(recs, 5, rng_for(1, "x"), SPLITS)


def test_per_class_prefers_test_split():
    recs = [rec(["a"], "train") for _ in range(10)] + [rec(["a"], "test") for _ in range(5)]
    picked = select_per_class(recs, 5, rng_for(0, "x"), SPLITS)
    assert all(recs[i]["ref"]["split"] == "test" for i in picked)


def test_greedy_cover_reaches_quota_without_duplicates():
    recs = [rec(["fruit"]) for _ in range(30)] + [rec(["fruit", "flower"]) for _ in range(3)] + [rec(["leaf"]) for _ in range(8)]
    picked = select_greedy_cover(recs, 5, rng_for(0, "x"), SPLITS)
    assert len(picked) == len(set(picked))
    have = Counter(k for i in picked for k in set(recs[i]["keys"]))
    assert have["flower"] == 3 and have["fruit"] >= 5 and have["leaf"] == 5


def test_itt_round_robin_covers_every_stratum():
    recs = [rec([], task=t) for t in ["a"] * 100 + ["b"] * 2 + ["c"] * 50]
    picked, keys = select_itt(recs, ITT_CFG, rng_for(0, "x"), SPLITS)
    assert keys == ["task"]
    assert Counter(recs[i]["strata"]["task"] for i in picked) == {"a": 5, "b": 2, "c": 5}


def test_itt_fill_to_cap():
    recs = [rec([], task=t) for t in ["a"] * 100 + ["b"] * 100]
    picked, _ = select_itt(recs, {**ITT_CFG, "fill_to_cap": True}, rng_for(0, "x"), SPLITS)
    assert len(picked) == 40


def test_qa_format():
    assert qa_format("Which?\nA. x\nB. y", "A. x") == "multiple_choice"
    assert qa_format("Is it?", "Yes.") == "yes_no"
    assert qa_format("How many?", "12") == "numeric"
    assert qa_format("Describe.", "A leaf.") == "open_ended"


def test_bbox_conversion():
    assert to_pixel_xywh([10, 20, 30, 40], (100, 100), None) == [10, 20, 30, 40]
    assert to_pixel_xywh([10, 20, 40, 60], (100, 100), "xyxy") == [10, 20, 30, 40]
    assert to_pixel_xywh([0.5, 0.5, 0.2, 0.4], (100, 200), None) == [40.0, 60.0, 20.0, 80.0]


def test_mask_class_map():
    recs = [{"keys": ["0", "1", "2"]}]
    assert mask_class_map(recs, ["background", "crop", "weed"]) == {"0": "background", "1": "crop", "2": "weed"}
    assert mask_class_map(recs, ["crop", "weed"]) == {"0": "background", "1": "crop", "2": "weed"}
    assert mask_class_map([{"keys": ["0", "255"]}], ["fruit"]) == {"0": "background", "255": "fruit"}
