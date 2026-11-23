import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from agbench.eval import Benchmark, Method, VLMMethod
from agbench.eval import scoring as sc
from agbench.eval.agtask import class_bucket, itt_bucket, split_classes
from agbench.eval.crops import phrase, resolve
from agbench.eval.items import N_OPTIONS, build_items, mask_boxes
from agbench.eval.judge import judge_run, parse_verdict
from agbench.eval.resources import prompts

NOTA, IDK = prompts()["nota"], prompts()["idk"]


def _dataset(root, task, name, samples, catalog, info=None, images=(), masks=()):
    d = root / "base-data" / task / name
    (d / "images").mkdir(parents=True)
    for rel in images:
        Image.new("RGB", (100, 80), (90, 140, 60)).save(d / rel)
    for rel, arr in masks:
        (d / "masks").mkdir(exist_ok=True)
        Image.fromarray(arr).save(d / rel)
    (d / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    (d / "meta.json").write_text(json.dumps({"name": name, "catalog": catalog, "info": info or {}}))


def _cls(root, name, labels, crops, at="disease_classification", per=10):
    samples = [{"sample_id": f"{name}-{l}-{i}", "label": l, "images": [f"images/{l}-{i}.jpg"], "image_size": [100, 80]}
               for l in labels for i in range(per)]
    _dataset(root, "classification", name, samples, {"classes": labels, "crops": crops, "agricultural_task": at},
             images=[s["images"][0] for s in samples])


@pytest.fixture
def data_dir(tmp_path):
    _cls(tmp_path, "tomato_leaf_disease", ["Early_blight", "Late_blight", "healthy", "Leaf_Mold", "Spider Mites Two-spotted Spider Mite"], ["tomato"])
    _cls(tmp_path, "bean_cowpea_disease", ["bean_rust", "cowpea_mosaic", "angular_spot"], ["bean", "cowpea"])
    _cls(tmp_path, "weed_species", ["Ragweed", "Pigweed", "Crabgrass", "Foxtail", "Nutsedge"], ["maize"], at="weed_classification")
    _dataset(tmp_path, "detection", "apple_detection",
             [{"sample_id": "apple-0", "images": ["images/a0.jpg"], "image_size": [100, 80],
               "objects": [{"label": "apple", "bbox": [10, 10, 20, 20]}, {"label": "apple", "bbox": [50, 40, 30, 30]}]}],
             {"classes": ["apple"], "crops": ["apple"], "agricultural_task": "crop_detection"}, {"class_names": ["apple"]},
             images=["images/a0.jpg"])
    m = np.zeros((80, 100), np.uint8)
    m[10:30, 10:40] = 1
    m[50:70, 60:90] = 1
    m[0:1, 99:100] = 1  # below min area
    _dataset(tmp_path, "segmentation", "fruit_seg",
             [{"sample_id": "seg-0", "images": ["images/s0.jpg"], "image_size": [100, 80], "mask": "masks/s0.png"}],
             {"classes": [], "crops": ["apple"], "agricultural_task": "crop_segmentation"},
             {"mask_classes": {"0": "background", "1": "apple"}}, images=["images/s0.jpg"], masks=[("masks/s0.png", m)])
    _dataset(tmp_path, "image_text_to_text", "AgroBench", [
        {"sample_id": "ab-0", "images": ["images/q0.jpg"], "question": "What disease is this?\nA. Rust\nB. Blight\nC. Mildew",
         "answer": "B. Blight", "strata": {"source": "did", "qa_format": "multiple_choice", "crop": "Maize"}, "dataset": "AgroBench"},
        {"sample_id": "ab-1", "images": ["images/q0.jpg"], "question": "Is this leaf healthy?", "answer": "No",
         "strata": {"source": "pid", "qa_format": "yes_no"}, "dataset": "AgroBench"},
        {"sample_id": "ab-2", "images": ["images/q0.jpg"], "question": "How many apples?", "answer": "14",
         "strata": {"source": "cmn", "qa_format": "numeric"}, "dataset": "AgroBench"},
        {"sample_id": "ab-3", "images": ["images/q0.jpg"], "question": "Describe the symptoms.", "answer": "Orange pustules of rust.",
         "strata": {"source": "tm", "qa_format": "open_ended"}, "dataset": "AgroBench"},
    ], {"classes": [], "crops": ["maize"]}, images=["images/q0.jpg"])
    return tmp_path


def test_items_are_deterministic(data_dir):
    a, _ = build_items(data_dir, seed=0)
    b, _ = build_items(data_dir, seed=0)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    c, _ = build_items(data_dir, seed=1)
    assert [x.prompt for x in a] != [x.prompt for x in c]


def test_mcqa_options_capped_with_nota_and_idk(data_dir):
    items, _ = build_items(data_dir, tasks=["classification"], formats=["mcqa_within", "mcqa_across"])
    for it in items:
        assert it.choices[-2:] == [NOTA, IDK]
        assert len(it.choices) - 2 <= N_OPTIONS
        assert len(set(it.choices)) == len(it.choices)
        assert (it.answer == NOTA) == it.is_nota
        assert it.is_nota or it.answer in it.choices
    nota_share = sum(it.is_nota for it in items) / len(items)
    assert 0.2 < nota_share < 0.4


def test_across_distractors_from_other_datasets_same_bucket(data_dir):
    items, _ = build_items(data_dir, tasks=["classification"], formats=["mcqa_across"], only="tomato")
    tomato = {"Early blight", "Late blight", "healthy", "Leaf Mold", "Spider Mites Two-spotted Spider Mite"}
    bean = {"bean rust", "cowpea mosaic", "angular spot"}
    for it in items:
        distractors = [c for c in it.choices[:-2] if c != it.answer]
        assert not set(distractors) & tomato
        if it.ag_task == "disease":
            assert set(distractors) <= bean | {"Ragweed", "Pigweed", "Crabgrass", "Foxtail", "Nutsedge"}


def test_crop_resolution(data_dir):
    items, _ = build_items(data_dir, tasks=["classification"], formats=["oeq"])
    by = {it.dataset: it for it in items}
    assert by["tomato_leaf_disease"].crop == "tomato" and "a tomato plant" in by["tomato_leaf_disease"].prompt
    assert by["weed_species"].crop == "plant" and by["weed_species"].crop_source == "species_id_plant"
    bean = [it for it in items if it.dataset == "bean_cowpea_disease"]
    assert {it.crop for it in bean} >= {"bean", "cowpea"}  # matched from the label
    assert resolve({"crops": ["bean", "cowpea"]}, "disease", ["angular_spot"]) == ("bean or cowpea", "catalog_list")
    assert phrase("apple") == "an apple plant"


def test_ag_task_class_overrides():
    entry = {"agricultural_task": "disease_classification"}
    assert class_bucket("tomato_leaf_disease", entry, "Spider Mites Two-spotted Spider Mite") == "pest_damage"
    assert class_bucket("tomato_leaf_disease", entry, "Early Blight") == "disease"
    assert class_bucket("arabica_coffee_leaf_disease_classification", entry, "Miner") == "pest_damage"
    assert class_bucket("CT_PestNet_crop_pest_classification", {}, "rice borer") == "pest_damage"
    assert class_bucket("x", {"agricultural_task": "crop_detection"}, "apple") == "localization"
    assert itt_bucket({"dataset": "AgroBench", "strata": {"source": "pid"}}) == "pest_damage"
    assert itt_bucket({"dataset": "MIRAGE", "strata": {"category": "Plant Disease Identification"}}) == "disease"
    assert split_classes(["a, b, c"]) == ["a", "b", "c"]


def test_mask_boxes_drop_small_components(data_dir):
    boxes = mask_boxes(data_dir / "base-data/segmentation/fruit_seg/masks/s0.png", {"0": "background", "1": "apple"})
    assert sorted(b["bbox"] for b in boxes) == [[10.0, 10.0, 40.0, 30.0], [60.0, 50.0, 90.0, 70.0]]


def test_choice_matching_cascade():
    opts = ["Early blight", "Late blight", "Leaf Mold", NOTA, IDK]
    assert sc.match_choice("late blight", opts) == "Late blight"
    assert sc.match_choice("The answer is Leaf Mold.", opts) == "Leaf Mold"
    assert sc.match_choice("Erly blight", opts) == "Early blight"
    assert sc.match_choice("none of the above", opts) == NOTA
    assert sc.match_choice("banana", opts) is None
    assert sc.final_answer("<think>x</think>\n<answer>Rust</answer>") == "Rust"
    assert sc.parse_letters("B", 4) == ["B"] and sc.parse_letters("A, C", 4) == ["A", "C"]
    assert sc.parse_letters("A healthy leaf", 4) == []


def test_box_parsing_and_iou():
    boxes = sc.parse_boxes('```json\n[{"box_2d": [100, 200, 500, 600], "label": "apple"}]\n```', 1000, 1000)
    assert boxes == [{"label": "apple", "bbox": [200.0, 100.0, 600.0, 500.0]}]
    q = sc.parse_boxes('[{"bbox_2d": [200, 100, 600, 500]}]', 1000, 1000, {"bbox_2d": "xyxy_1000"})
    assert q[0]["bbox"] == [200.0, 100.0, 600.0, 500.0]
    assert sc.parse_boxes("[]", 10, 10) == [] and sc.parse_boxes("no boxes", 10, 10) is None
    assert sc.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    m = sc.match_boxes([{"label": "a", "bbox": [0, 0, 10, 10]}], [{"label": "a", "bbox": [0, 0, 10, 10]}, {"label": "a", "bbox": [50, 50, 60, 60]}], ["a"])
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1) and m["acc"] == 0.5


class Oracle(Method):
    name = "oracle"

    async def predict(self, it):
        if it.gt_boxes is not None:
            w, h = it.image_size
            return json.dumps([{"box_2d": [b["bbox"][1] / h * 1000, b["bbox"][0] / w * 1000, b["bbox"][3] / h * 1000, b["bbox"][2] / w * 1000],
                                "label": b["label"]} for b in it.gt_boxes])
        if it.qa_format == "multiple_choice":
            return ",".join(it.answer)
        return str(it.answer)


def test_benchmark_oracle_scores_perfectly_and_resumes(data_dir, tmp_path):
    bench = Benchmark(data_dir)
    res = bench.run(Oracle(), tmp_path / "run")
    assert all(r["accuracy_macro"] == 1.0 and r["f1_macro"] == 1.0 for r in res.metrics["cv_task"])
    assert {r["cv_task"] for r in res.metrics["cv_task"]} == {
        "classification/mcqa_within", "classification/mcqa_across", "classification/oeq", "detection", "segmentation", "image_text_to_text"}
    assert any(r["ag_task"] == "pest_damage" for r in res.metrics["ag_task"])
    n = len((tmp_path / "run" / "predictions.jsonl").read_text().splitlines())
    assert bench.remaining(tmp_path / "run") == []

    class Boom(Method):
        name = "boom"

        async def predict(self, it):
            raise AssertionError("resume should skip finished items")

    bench.run(Boom(), tmp_path / "run")
    assert len((tmp_path / "run" / "predictions.jsonl").read_text().splitlines()) == n


class FakeOpenAI(BaseHTTPRequestHandler):
    """Answers like a thinking VLM: reasoning separate, final answer in content; judge requests get JSON."""
    requests = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append(body)
        text = body["messages"][-1]["content"]
        text = text if isinstance(text, str) else next(c["text"] for c in text if c["type"] == "text")
        if "Respond with JSON only" in text:
            answer = '{"is_match": true, "reasoning": "same"}'
        elif "Respond with ONLY the category name" in text:
            answer = "Late blight"
        else:
            answer = "rust"
        self._send({"choices": [{"message": {"content": answer, "reasoning_content": "thinking..."}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_vlm_method_and_judge_against_fake_server(data_dir, tmp_path, fake_server):
    FakeOpenAI.requests.clear()
    bench = Benchmark(data_dir, tasks=["classification"], only="tomato")
    res = bench.run(VLMMethod(fake_server, "google/gemma-4-12B-it", effort="medium"), tmp_path / "r", concurrency=4)
    body = FakeOpenAI.requests[0]
    assert body["chat_template_kwargs"]["enable_thinking"] is True and body["top_k"] == 64
    assert body["messages"][-1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    recs = res.records
    assert all(r["reasoning"] == "thinking..." for r in recs)
    assert res.metrics["pending_judge"] > 0
    metrics = judge_run(tmp_path / "r", fake_server, "RedHatAI/gemma-4-31B-it-FP8-dynamic")
    assert metrics["pending_judge"] == 0
    oeq = next(r for r in metrics["cv_task"] if r["cv_task"] == "classification/oeq")
    assert oeq["accuracy_macro"] == 1.0  # fake judge says every answer matches
    judge_body = FakeOpenAI.requests[-1]
    assert judge_body["temperature"] == 0.0 and judge_body["chat_template_kwargs"]["enable_thinking"] is False
    n = len(FakeOpenAI.requests)
    judge_run(tmp_path / "r", fake_server, "RedHatAI/gemma-4-31B-it-FP8-dynamic")
    assert len(FakeOpenAI.requests) == n  # nothing left to judge
    judge_run(tmp_path / "r", fake_server, "RedHatAI/gemma-4-31B-it-FP8-dynamic", redo=True)
    assert len(FakeOpenAI.requests) == n + sum(bool(r.get("needs_judge")) for r in recs)  # everything re-judged

    sessions = json.loads((tmp_path / "r" / "compute.json").read_text())
    assert [x["phase"] for x in sessions] == ["generate", "judge", "judge"] and sessions[0]["items"] == len(recs)
    from agbench.eval.profile import actual_frames
    from agbench.eval.report import load_runs
    prof = {"models": [{"model": "google/gemma-4-12B-it", "load_s": 60,
                        "tasks": {k: {"batch": {"throughput_per_s": 2.0}} for k in ("classification/mcqa_within", "classification/mcqa_across", "classification/oeq")}}],
            "judge": {"throughput_per_s": 5.0}}
    act = actual_frames(load_runs([tmp_path / "r"]), prof)
    row = act["models"].iloc[0]
    assert row["items"] == len(recs) and row["judged"] == sessions[1]["items"]
    assert row["generate h (est.)"] == round(len(recs) / 2.0 / 3600, 2)
    assert set(act["tasks"].task) == {"classification/mcqa_within", "classification/mcqa_across", "classification/oeq"}


def test_cot_template_for_models_without_thinking(fake_server):
    m = VLMMethod(fake_server, "mistralai/Ministral-3-14B-Instruct-2512")
    assert m.system and "<answer>" in m.system
    assert VLMMethod(fake_server, "google/gemma-4-12B-it").system is None


def test_rouge_l():
    assert sc.rouge_l("brown spots on the leaves", "brown spots on the leaves") == 1.0
    assert sc.rouge_l("", "x") == 0.0 and sc.rouge_l("green", "brown") == 0.0
    assert sc.rouge_l("the leaf has brown spots", "brown spots scattered on the leaf") == pytest.approx(2 * (2 / 5) * (2 / 6) / (2 / 5 + 2 / 6))


def test_parse_verdict():
    assert parse_verdict('```json\n{"is_match": false, "reasoning": "x"}\n```') is False
    assert parse_verdict('is_match: true') is True
    assert parse_verdict("garbage") is None


def test_register_models_from_yaml(tmp_path, monkeypatch):
    import copy

    from agbench.eval import register_models
    from agbench.eval.methods import model_spec, request_body
    from agbench.eval.resources import load
    from agbench.eval.servers import command

    monkeypatch.setattr("agbench.eval.servers._bin", lambda env, default: default)
    reg = load("models.yaml")
    saved = copy.deepcopy(reg)
    try:
        f = tmp_path / "mine.yaml"
        f.write_text("my-org/My-VLM:\n  serve_args: [--quantization, awq]\n  sampling: {temperature: 0.2}\n"
                     "  thinking: {chat_template_kwargs: {enable_thinking: true}}\n")
        register_models(f)
        spec = model_spec("my-org/My-VLM")
        assert spec["backend"] == "vllm" and spec["max_tokens"] == reg["defaults"]["max_tokens"]
        assert request_body(spec, True, "medium") == {"temperature": 0.2, "chat_template_kwargs": {"enable_thinking": True}}
        cmd = command("my-org/My-VLM", 8000)
        assert cmd[:3] == ["vllm", "serve", "my-org/My-VLM"] and cmd[-2:] == ["--quantization", "awq"]
        assert command("/models/unregistered", 8000)[2] == "/models/unregistered"
        nemo = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF"
        register_models({nemo: {"parallel": 4}})
        assert model_spec(nemo)["parallel"] == 4 and model_spec(nemo)["backend"] == "llamacpp" and model_spec(nemo)["gguf"]
    finally:
        reg.clear(); reg.update(saved)


def test_steady_wall_ignores_ramp_down_tail():
    import heapq
    import random
    from agbench.eval.profile import steady_wall
    rng = random.Random(0)
    lat = [rng.choice([5.0, 10.0, 20.0, 200.0]) for _ in range(300)]
    slots, recs = [0.0] * 32, []
    for x in lat:
        start = heapq.heappop(slots)
        heapq.heappush(slots, start + x)
        recs.append({"latency_s": x, "t_end": start + x})
    ideal, wall = sum(lat) / 32, max(r["t_end"] for r in recs)
    assert wall > ideal * 1.2  # the ramp-down tail inflates the raw wall time
    assert abs(steady_wall(recs, 32, wall) - ideal) / ideal < 0.1


def test_box_json_after_prose_coordinates():
    text = 'The plant is at [12, 16, 185, 300].\n\n```json\n[{"box_2d": [100, 200, 300, 400], "label": "maize"}]\n```'
    assert sc.parse_boxes(text, 1000, 1000) == [{"label": "maize", "bbox": [200.0, 100.0, 400.0, 300.0]}]
    assert sc.parse_boxes('see [1, 2, 3, 4] then [{"box_2d": [0, 0, 500, 500]}]', 1000, 1000)[0]["bbox"] == [0, 0, 500, 500]


def test_box_order_switches_prompt_and_parser(data_dir, fake_server):
    from agbench.eval.methods import key_formats, model_spec
    it = next(i for i in build_items(data_dir, ["detection"])[0])
    xy = VLMMethod(fake_server, "mistralai/Ministral-3-14B-Instruct-2512")
    yx = VLMMethod(fake_server, "google/gemma-4-12B-it")
    assert "[x_min, y_min, x_max, y_max]" in xy.prompt(it) and "[y_min" not in xy.prompt(it)
    assert yx.prompt(it) == it.prompt and "[y_min, x_min, y_max, x_max]" in it.prompt
    text = '[{"box_2d": [100, 200, 300, 400], "label": "apple"}]'
    assert sc.parse_boxes(text, 1000, 1000, key_formats(model_spec(xy.model_id)))[0]["bbox"] == [100, 200, 300, 400]
    assert sc.parse_boxes(text, 1000, 1000, key_formats(model_spec(yx.model_id)))[0]["bbox"] == [200, 100, 400, 300]


def test_profile_effort_variants_load_once(data_dir, tmp_path, fake_server, monkeypatch):
    import contextlib
    from agbench.eval import profile as pf
    loads = []

    @contextlib.contextmanager
    def server(model_id, log_path, **kw):
        loads.append(model_id)
        yield type("S", (), {"endpoint": fake_server, "load_seconds": 1.0})()

    class GPU:
        def summary(self, *a):
            return {}

    monkeypatch.setattr(pf, "Server", server)
    items, _ = build_items(data_dir, ["classification"], ["mcqa_within"])
    FakeOpenAI.requests.clear()
    rows = pf.profile_model("google/gemma-4-12B-it", items, tmp_path / "p", 2, 0, None, True, "medium", GPU(),
                            ["off", "low", "high"])
    assert loads == ["google/gemma-4-12B-it"] and [r["model"] for r in rows] == [f"google/gemma-4-12B-it · {v}" for v in ("off", "low", "high")]
    bodies = FakeOpenAI.requests
    assert bodies[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert any(b.get("thinking_token_budget") == 1024 for b in bodies) and not bodies[-1].get("thinking_token_budget")
    assert pf.variant_settings(pf.model_spec("Qwen/Qwen3.8-27B-FP8"), "low") == (True, "low", {})
    assert all("tasks" in r and not r.get("error") for r in rows)


def test_parse_letters_ignores_lowercase_words():
    assert sc.parse_letters("The answer is a weevil, not a beetle.", 4) == []
    assert sc.parse_letters("The answer is B.", 4) == ["B"]
    assert sc.parse_letters("answer: c", 4) == []
    assert sc.parse_letters("A, C", 4) == ["A", "C"]


def test_mcqa_ablation_options(data_dir):
    from agbench.eval.items import MCQAOptions
    mc = lambda **kw: {it.item_id: it for it in build_items(data_dir, ["classification"], ["mcqa_within", "mcqa_across"], mcqa=MCQAOptions(**kw) if kw else None)[0]}
    base = mc()
    assert base == mc(nota_rate=0.3, order_seed=None, nota_position="end", idk=True)  # defaults unchanged
    shuffled = mc(order_seed=1)
    for k, it in base.items():  # same options and NOTA items, only the order changes
        assert set(shuffled[k].choices) == set(it.choices) and shuffled[k].is_nota == it.is_nota and shuffled[k].answer == it.answer
        assert shuffled[k].choices[-2:] == [NOTA, IDK]
    assert any(shuffled[k].choices != base[k].choices for k in base)
    rand = mc(order_seed=1, nota_position="random")
    assert any(it.choices.index(NOTA) != len(it.choices) - 2 for it in rand.values())
    lo, hi = mc(nota_rate=0.1), mc(nota_rate=0.5)
    assert {k for k, it in lo.items() if it.is_nota} <= {k for k, it in base.items() if it.is_nota} <= {k for k, it in hi.items() if it.is_nota}
    assert all(IDK not in it.choices and IDK not in it.prompt for it in mc(idk=False).values())


def test_gen_seed_is_sent_per_item(data_dir, fake_server, tmp_path):
    from agbench.eval.methods import item_seed
    FakeOpenAI.requests.clear()
    items = build_items(data_dir, ["classification"], ["mcqa_within"])[0][:3]
    Benchmark(items=items).run(VLMMethod(fake_server, "google/gemma-4-12B-it", gen_seed=7), tmp_path / "r", concurrency=1, resume=False)
    assert sorted(b["seed"] for b in FakeOpenAI.requests) == sorted(item_seed(7, it.item_id) for it in items)
    assert item_seed(7, "x") == item_seed(7, "x") != item_seed(8, "x")


def test_stats_bootstrap_and_tests():
    from agbench.eval import stats as S
    a = {f"d{i}": 0.6 + 0.01 * (i % 3) for i in range(30)}
    b = {k: v - 0.05 for k, v in a.items()}  # A beats B by 5 points on every dataset
    p = S.paired(a, b)
    assert abs(p["diff"] - 0.05) < 1e-9 and p["lo"] > 0 and p["p_boot"] < 0.01
    tied = S.paired(a, {k: v + (0.05 if i % 2 else -0.05) for i, (k, v) in enumerate(a.items())})
    assert tied["lo"] < 0 < tied["hi"]
    c = S.ci(a)
    assert c["lo"] <= c["mean"] <= c["hi"]
    assert S.mcnemar({1: True, 2: True, 3: False}, {1: False, 2: True, 3: False}) == {"a_only": 1, "b_only": 0, "p": 1.0}
    assert S.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    recs = [{"dataset": "d", "ag_task": "disease", "is_nota": n, "nota_pick": pick, "idk": False, "acc": float(n == pick)}
            for n, pick in ((True, True), (True, False), (False, True), (False, False))]
    beh = S.behaviour(recs)
    assert beh["nota_recall"] == {"d": 0.5} and beh["false_nota"] == {"d": 0.5} and beh["over_commit"] == {"d": 0.5}
