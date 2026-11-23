import json

import pytest

from agbench import bench
from agbench.eval.baselines import NaiveBaseline, priors
from agbench.eval.methods import Method
from agbench.eval.runner import Benchmark
from agbench.viz import aggregate, html

from test_bench import _fake_dataset


class Letter(Method):
    def __init__(self, gen_seed=None):
        self.name, self.gen_seed = "toy/letter-a", gen_seed

    def predict(self, item):
        return "A"

    def config(self):
        return {"name": self.name, "gen_seed": self.gen_seed}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("viz")
    full = bench.load("agml-full-v1")
    for ds in ("bean_disease_uganda", "tomato_leaf_disease"):
        _fake_dataset(tmp, full, ds)
    b = Benchmark(tmp, tasks=["classification"], limit_per_dataset=4)
    for s in (0, 1):
        b.run(Letter(s), tmp / "runs" / f"letter_s{s}")
    b.run(NaiveBaseline("chance", priors(b.items), 0), tmp / "runs" / "chance")
    return tmp / "runs"


def test_entries_group_seeds_and_kinds(runs):
    es = aggregate.entries([runs], labels={"toy/letter-a": "Letter A"})
    assert [(e.label, e.kind, e.replicates) for e in es] == [("Letter A", "model", 2), ("baseline/chance", "chance", 1)]
    e = es[0]
    m, lo, hi, n = e.cells["mcqa_within"]
    assert n == 2 and lo <= m <= hi and e.cells["detection"][3] == 0
    per = [e.datasets[("mcqa_within", d)] for d in ("bean_disease_uganda", "tomato_leaf_disease")]
    assert m == pytest.approx(sum(per) / 2)


def test_bootstrap_exact_matches_fast_closely():
    vals = [10.0, 20.0, 30.0, 45.0, 60.0, 5.0]
    fast, exact = aggregate.boot_ci(vals), aggregate.boot_ci(vals, exact=True)
    assert all(abs(a - b) < 3 for a, b in zip(fast, exact))


def test_html_report_is_self_contained(runs, tmp_path):
    p = html.write(aggregate.entries([runs]), tmp_path / "r.html", "Toy </script> report")
    page = p.read_text()
    assert "<script src" not in page and "http://" not in page.replace("http://www.w3.org/2000/svg", "")
    data = json.loads(page.split("const D = ", 1)[1].split(";\n", 1)[0])
    assert data["title"] == "Toy </script> report" and len(data["entries"]) == 2 and data["datasets"]
    assert "Toy &lt;/script&gt; report" in page.split("<title>")[1]


def test_figures(runs, tmp_path):
    pytest.importorskip("matplotlib")
    from agbench.viz.figures import agtask, main_results

    es = aggregate.entries([runs])
    assert main_results(es, tmp_path / "m.png").stat().st_size > 1000
    assert agtask(es, tmp_path / "a.png").stat().st_size > 1000
