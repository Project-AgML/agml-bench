from agml_bench.catalog import build_catalog, diff_catalogs
from agml_bench.config import load_config

CFG = load_config()


def site(entries):
    return {"hf_datasets.json": entries, "datasets.json": [], "dataset_history.json": {}}


def entry(name, task="image_classification", sensor="rgb", repo=None):
    return {"name": name, "machine_learning_task": task, "sensor_modality": sensor,
            "hf_link": f"https://huggingface.co/datasets/Project-AgML/{repo or name}"}


def hf(*names, tags=()):
    return {f"Project-AgML/{n}": {"sha": "s1", "last_modified": None, "tags": list(tags)} for n in names}


def test_eligibility_rules():
    cat = build_catalog(CFG, site([entry("a"), entry("b", sensor="multispectral"), entry("c", task="unlabeled"),
                                   entry("d")]), hf("a", "b", "c"), "sha")
    ds = cat["datasets"]
    assert ds["a"]["eligible"]
    assert ds["b"]["exclusion_reason"] == "sensor:multispectral"
    assert ds["c"]["exclusion_reason"] == "task:unlabeled"
    assert ds["d"]["exclusion_reason"] == "hf_repo_not_found"


def test_hf_only_itt_repos_are_added():
    cat = build_catalog(CFG, site([entry("a")]), {**hf("a"), **hf("NewVQA", tags=["task_categories:image-text-to-text"])}, "sha")
    assert cat["datasets"]["NewVQA"]["task"] == "image_text_to_text"
    assert cat["datasets"]["NewVQA"]["eligible"]


def test_shared_link_is_repaired():
    cat = build_catalog(CFG, site([entry("sunflower"), entry("carrot", repo="sunflower")]), hf("sunflower", "carrot"), "sha")
    assert cat["datasets"]["carrot"]["hf_repo"] == "Project-AgML/carrot"


def test_diff_and_first_seen():
    old = build_catalog(CFG, site([entry("a"), entry("b")]), hf("a", "b"), "sha")
    old["datasets"]["a"]["first_seen"] = "2020-01-01"
    new_hf = hf("a", "c")
    new_hf["Project-AgML/a"]["sha"] = "s2"
    new = build_catalog(CFG, site([entry("a"), entry("c")]), new_hf, "sha", old)
    d = diff_catalogs(old, new)
    assert d["added"] == ["c"] and d["removed"] == ["b"] and d["changed"] == {"a": ["hf_sha"]}
    assert new["datasets"]["a"]["first_seen"] == "2020-01-01"
