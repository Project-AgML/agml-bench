import json

import pytest

from agbench import bench
from agbench.eval.items import MCQAOptions
from agbench.eval.runner import Benchmark


@pytest.fixture(scope="module")
def full():
    return bench.load("agml-full-v1")


def _fake_dataset(data_dir, b, name):
    """Mark one bundle dataset as downloaded: meta.json at the pinned revision and its sample ids."""
    m = b.manifests[name]
    d = data_dir / "base-data" / b.raw_items[[r["dataset"] for r in b.raw_items].index(name)]["task"] / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"name": name, "hf_sha": m["hf_sha"], "catalog": m["catalog"]}))
    (d / "samples.jsonl").write_text("".join(json.dumps({"sample_id": s["sample_id"], "dataset": name, **s["ann"],
                                                         "image": f"images/{s['sample_id']}.jpg"}) + "\n" for s in m["samples"]))
    return d


def test_bundle_files_match_hash(full):
    assert full.check() == []
    assert full.spec["items"]["total"] == len(full.raw_items) == 32456
    assert full.spec["datasets"]["total"] == len(full.manifests) == 372
    assert {r["dataset"] for r in full.raw_items} == set(full.manifests)
    assert all(not p.startswith("/") for r in full.raw_items for p in r["images"])


def test_unknown_bundle():
    with pytest.raises(KeyError):
        bench.load("agml-nope-v0")


def test_items_only_for_downloaded_datasets(full, tmp_path):
    _fake_dataset(tmp_path, full, "bean_disease_uganda")
    items, skipped = full.items(tmp_path)
    assert items and {it.dataset for it in items} == {"bean_disease_uganda"}
    assert all(it.images[0].startswith(str(tmp_path)) for it in items)
    assert len(skipped) == 371 and "agbench data pull" in skipped[0]["reason"]
    within, _ = full.items(tmp_path, formats=["mcqa_within"], limit_per_dataset=3)
    assert len(within) == 3 and {it.format for it in within} == {"mcqa_within"}


def test_benchmark_records_bundle_and_variant(full, tmp_path, monkeypatch):
    _fake_dataset(tmp_path, full, "bean_disease_uganda")
    std = Benchmark(tmp_path, tasks=["classification"])
    assert std.args["benchmark"] == {"name": "agml-full-v1", "hash": full.hash, "variant": None}
    assert [it.to_dict()["prompt"] for it in std.items][:1] == [r["prompt"] for r in full.raw_items if r["dataset"] == "bean_disease_uganda"][:1]
    seen = {}
    monkeypatch.setattr("agbench.eval.runner.build_items", lambda *a: (seen.setdefault("only", a[3]), [])[1:] + ([],))
    abl = Benchmark(tmp_path, tasks=["classification"], only="bean", mcqa=MCQAOptions(nota_rate=0.5))
    assert abl.args["benchmark"]["variant"] == "ablation"
    import re
    assert re.search(seen["only"], "bean_disease_uganda") and not re.search(seen["only"], "bean_disease_uganda_v2")
    assert not re.search(seen["only"], "tomato_leaf_disease")  # in the bundle, but excluded by --only


def test_data_status_and_verify(full, tmp_path):
    d = _fake_dataset(tmp_path, full, "bean_disease_uganda")
    st = bench.data_status(full, tmp_path, only="^bean_disease")
    assert st == {"bean_disease_classification_tanzania": "missing", "bean_disease_uganda": "done"}
    problems = bench.verify(full, tmp_path, only="^bean_disease_uganda$", decode=False)
    assert any("missing image" in e for e in problems["bean_disease_uganda"])
    (d / "meta.json").write_text(json.dumps({"hf_sha": "0" * 40}))
    assert bench.data_status(full, tmp_path, only="^bean_disease_uganda$") == {"bean_disease_uganda": "stale"}


def test_failure_reasons_are_actionable():
    from agbench.pipeline import explain

    repo, rev = "Project-AgML/X", "c421bedca8dfa4f33ddd07cca19365dfea4a9657"
    gone = FileNotFoundError(f"datasets/{repo}@{rev}/train.parquet (repository not found)")
    assert explain(gone, repo, rev) == f"source repo {repo} not found (deleted, renamed or private)"
    moved = FileNotFoundError(f"datasets/{repo}@{rev}/train.parquet (revision not found)")
    assert "pinned revision c421bed no longer exists" in explain(moved, repo, rev)
    assert "read-only" in explain(RuntimeError("403 Forbidden: You have read access"), repo, rev)
    assert explain(ValueError("boom"), repo, rev) == "ValueError: boom"
