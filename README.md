# agml-bench

Benchmark vision-language models on [AgML](https://github.com/Project-AgML/AgML) agricultural datasets: classification, detection, segmentation and VQA.

| Benchmark | Datasets | Items | Hash |
|---|---|---|---|
| `agml-full-v1` | 372 | 32,456 | `1876e21c16b413dd` |

AgriBench (in `agml-full-v1`) was removed from Hugging Face; `data pull` skips it and runs report it as unavailable.

## Install

```bash
git clone https://github.com/Project-AgML/agml-bench.git && cd agml-bench
uv sync
hf auth login
uv tool install vllm          # to serve models with --model
```

Without a clone:

```bash
uv tool install "agml-bench[viz]"        # or: pip install "agml-bench[viz]"
agml-bench bench list
```

## Quick start

```bash
DATA=/path/to/data_dir

uv run agml-bench data pull agml-full-v1 --data-dir $DATA
uv run agml-bench eval run --model my-org/My-VLM --data-dir $DATA --judge
uv run agml-bench eval report --runs runs
```

## Run your model

Hugging Face repo or local checkpoint (served with vLLM):

```bash
uv run agml-bench eval run --model my-org/My-VLM --data-dir $DATA --judge
uv run agml-bench eval run --model /path/to/checkpoint --data-dir $DATA --judge
```

Already served (any OpenAI-compatible endpoint):

```bash
uv run agml-bench eval run --model my-org/My-VLM --endpoint http://localhost:8000 --data-dir $DATA --judge
```

Hosted API (OpenAI-compatible):

```bash
export OPENAI_API_KEY=...
uv run agml-bench eval run --model gpt-5-mini --endpoint https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --concurrency 8 --data-dir $DATA --judge
uv run agml-bench eval run --model gemini-2.5-flash --endpoint https://generativelanguage.googleapis.com/v1beta/openai --api-key-env GEMINI_API_KEY --data-dir $DATA --judge
uv run agml-bench eval run --model claude-sonnet-5-5 --endpoint https://api.anthropic.com/v1 --api-key-env ANTHROPIC_API_KEY --data-dir $DATA --judge
```

Model settings:

```bash
uv run agml-bench eval run --model my-org/My-VLM --data-dir $DATA --models-file my_models.yaml --judge
```

```yaml
# my_models.yaml (fields as in src/agml_bench/eval/models.yaml)
my-org/My-VLM:
  serve_args: [--max-model-len, "32768"]
  sampling: {temperature: 0.7, top_p: 0.95}
  thinking: {chat_template_kwargs: {enable_thinking: true}}
  no_thinking: {chat_template_kwargs: {enable_thinking: false}}
  effort: {low: {reasoning_effort: low}, high: {reasoning_effort: high}}   # request fields per --reasoning-effort
  box_order: xyxy                       # box_2d as [x_min, y_min, x_max, y_max]; default yxyx
  bbox_key_format: {bbox_2d: xyxy_px}   # other box keys: xyxy_1000 | xyxy_px
```

Your own method:

```bash
uv run agml-bench eval run --method mypkg.methods:MyMethod --data-dir $DATA
```

```python
from agml_bench.eval import Benchmark, EvalItem, Method

class MyMethod(Method):
    name = "my-method"
    async def predict(self, item: EvalItem) -> str:
        ...  # item.images, item.prompt, item.choices, item.task, item.format

res = Benchmark(DATA, bench="agml-full-v1", tasks=["classification"]).run(MyMethod(), out_dir="runs/my-method")
res.metrics
```

## Options

```bash
--task classification          # classification | detection | segmentation | image_text_to_text
--format mcqa_within           # mcqa_within | mcqa_across | oeq | grounding | multiple_choice
--only '^bean_disease_uganda$' # regex on dataset name
--limit-per-dataset 2          # quick try
--no-thinking
--reasoning-effort medium      # low (default) | medium | high
--concurrency 16
--gen-seed 1                   # sampling seed (replicate runs)
--run-dir runs/<run_dir>       # resume
--benchmark agml-full-v1       # default; 'none' builds items from $DATA
--strict                       # exit non-zero if the run check fails or the run is not comparable
```

## Results

```bash
uv run agml-bench eval check runs/<run_dir>                        # files, schema, items vs benchmark, coverage, comparable
uv run agml-bench eval judge --run runs/<run_dir>
uv run agml-bench eval judge --run runs/<run_dir> --judge-model gpt-5-mini --judge-endpoint https://api.openai.com/v1 --judge-api-key-env OPENAI_API_KEY   # not comparable
uv run agml-bench eval report --runs runs
uv run agml-bench eval report --runs runs --csv reports/scores     # scores.csv, ag_tasks.csv, datasets.csv
uv run agml-bench eval report --runs runs --html reports/report.html                       # charts, tables, runs; one offline file
uv sync --extra viz && uv run agml-bench eval report --runs runs --figures reports/figures   # paper figures (PNG)
uv run agml-bench eval report --runs runs --html reports/report.html --label "my-org/My-VLM=My VLM" --label "baseline/chance=Chance"
uv run agml-bench eval stats --runs runs --out reports/stats
uv run agml-bench eval view --runs runs                            # http://localhost:8502
```

`runs/<run_dir>/run_manifest.json`: agml-bench version, benchmark name and hash, model and sampling, judge, coverage, `comparable`.

## Benchmarks

```bash
uv run agml-bench bench list
uv run agml-bench bench show agml-full-v1
uv run agml-bench data pull agml-full-v1 --data-dir $DATA --task classification
uv run agml-bench data status agml-full-v1 --data-dir $DATA
uv run agml-bench data verify agml-full-v1 --data-dir $DATA
```

## Baselines

```bash
uv run agml-bench eval baseline --kind chance --data-dir $DATA --out runs/baselines/chance
```

```bash
uv sync --extra yolo
uv run agml-bench sample-train --data-dir $DATA --from-manifest
uv run --extra yolo agml-bench eval yolo-shared --data-dir $DATA --shots 10 --subset 0
```

## More

- New benchmark versions, catalog, sampling, profiling: [docs/building.md](https://github.com/Project-AgML/agml-bench/blob/main/docs/building.md)
- Paper experiments: [research/README.md](https://github.com/Project-AgML/agml-bench/blob/main/research/README.md)
- Evaluation rules: [docs/benchmarking.md](https://github.com/Project-AgML/agml-bench/blob/main/docs/benchmarking.md)

## Tests

```bash
uv run pytest
```

## License

Code: Apache-2.0 ([LICENSE](https://github.com/Project-AgML/agml-bench/blob/main/LICENSE)). Images and labels: each dataset's own license (`license` in [catalog/catalog.json](https://github.com/Project-AgML/agml-bench/blob/main/catalog/catalog.json)).
