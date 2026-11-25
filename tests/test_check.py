import asyncio
import json
import shutil

import httpx
import pytest

from agml_bench import bench
from agml_bench.eval.check import check_run
from agml_bench.eval.client import Client, FatalAPIError, api_base, api_key_from_env
from agml_bench.eval.methods import Method
from agml_bench.eval.runner import Benchmark

from test_bench import _fake_dataset


def test_api_base():
    assert api_base("http://localhost:8000") == "http://localhost:8000/v1"
    assert api_base("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert api_base("https://generativelanguage.googleapis.com/v1beta/openai/") == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_api_key_from_env(monkeypatch):
    assert api_key_from_env(None) == "EMPTY"
    monkeypatch.delenv("AGML_BENCH_TEST_KEY", raising=False)
    with pytest.raises(FatalAPIError, match="AGML_BENCH_TEST_KEY is not set"):
        api_key_from_env("AGML_BENCH_TEST_KEY")
    monkeypatch.setenv("AGML_BENCH_TEST_KEY", "sk-test")
    assert api_key_from_env("AGML_BENCH_TEST_KEY") == "sk-test"


def _client(handler, **kw) -> Client:
    from openai import AsyncOpenAI

    c = Client("https://api.example.com/v1", "gpt-x", api_key="sk-test", retries=1, **kw)
    c.api = AsyncOpenAI(base_url="https://api.example.com/v1", api_key="sk-test", max_retries=0,
                        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return c


def _ok(text="A"):
    return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "gpt-x",
                                     "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                                     "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})


def test_client_adapts_to_hosted_api_parameters():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}})
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.0 with this model."}})
        return _ok("B")

    c = _client(handler)
    p = asyncio.run(c.chat("q", [], max_tokens=64, body={"temperature": 0.0}))
    assert p.text == "B" and p.error is None
    assert seen[-1]["max_completion_tokens"] == 64 and "temperature" not in seen[-1] and len(seen) == 3


def test_client_bad_key_stops_the_run(tmp_path):
    c = _client(lambda req: httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}}))
    with pytest.raises(FatalAPIError, match="API key rejected"):
        asyncio.run(c.chat("q", []))

    class M(Method):
        name = "m"

        async def predict(self, item):
            return await c.chat(item.prompt, [])

    full = bench.load("agml-full-v1")
    _fake_dataset(tmp_path, full, "bean_disease_uganda")
    b = Benchmark(tmp_path, tasks=["classification"], formats=["mcqa_within"], limit_per_dataset=2)
    with pytest.raises(FatalAPIError):
        b.run(M(), tmp_path / "run")


class Letter(Method):
    name = "letter-a"

    def predict(self, item):
        return "A"


@pytest.fixture
def run_dir(tmp_path):
    full = bench.load("agml-full-v1")
    _fake_dataset(tmp_path, full, "bean_disease_uganda")
    b = Benchmark(tmp_path, tasks=["classification"], formats=["mcqa_within", "mcqa_across"], limit_per_dataset=3)
    b.run(Letter(), tmp_path / "run")
    return tmp_path / "run"


def test_manifest_and_check(run_dir):
    m = json.loads((run_dir / "run_manifest.json").read_text())
    assert m["benchmark"]["name"] == "agml-full-v1" and m["comparable"] and m["coverage"]["complete"] is False
    c = check_run(run_dir)
    assert c["errors"] == [] and c["stats"]["comparable"] and any("benchmark items" in w for w in c["warnings"])


def test_check_catches_edits(run_dir, tmp_path):
    bad = tmp_path / "edited"
    shutil.copytree(run_dir, bad)
    items = [json.loads(l) for l in open(bad / "items.jsonl")]
    items[0]["answer"] = "something else"
    (bad / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    recs = [json.loads(l) for l in open(bad / "predictions.jsonl")]
    recs[1]["correct"] = not recs[1]["correct"]
    (bad / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs + recs[:1]))
    errs = " | ".join(check_run(bad)["errors"])
    assert "differ from the frozen benchmark" in errs and "more than one prediction" in errs and "metrics.json does not match" in errs


def test_other_judge_not_comparable(run_dir):
    from agml_bench.eval.check import comparability

    run = json.loads((run_dir / "run.json").read_text())
    recs = [{"needs_judge": True, "judge_correct": True}]
    assert comparability({**run, "judge": {"model": "gpt-x", "prompt_sha": bench.judge_prompt_sha()}}, recs) == \
        ["judge gpt-x is not the reference judge RedHatAI/gemma-4-31B-it-FP8-dynamic"]
    ref = bench.load("agml-full-v1").spec["judge"]
    assert comparability({**run, "judge": {"model": ref["model"], "prompt_sha": ref["prompt_sha"]}}, recs) == []
    assert "not judged yet" in comparability(run, [{"needs_judge": True, "judge_correct": None}])[0]
