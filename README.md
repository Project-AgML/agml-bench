# AGgentic-benchmarks

Agricultural benchmarks for evaluating agentic systems.

## Setup

```bash
uv sync
hf auth login
```

## Catalog

```bash
uv run agbench catalog sync                 # fetch latest AgML catalog, print what changed
uv run agbench catalog diff                 # show upstream changes without saving
uv run agbench catalog list                 # eligible datasets
uv run agbench catalog list --all           # include excluded datasets with reason
uv run agbench catalog list --task image_text_to_text
```

## base-data

Sampling rules: [docs/subsampling.md](docs/subsampling.md)

```bash
DATA=/path/to/data_dir

uv run agbench status --data-dir $DATA
uv run agbench sample --data-dir $DATA
```

Filter:

```bash
uv run agbench sample --data-dir $DATA --task classification
uv run agbench sample --data-dir $DATA --only '^AgroBench$'
```

Update after a catalog sync:

```bash
uv run agbench catalog sync
uv run agbench sample --data-dir $DATA
```

Other options:

```bash
uv run agbench sample --data-dir $DATA --refresh          # resample datasets already done
uv run agbench sample --data-dir $DATA --skip-failed      # do not retry failed datasets
uv run agbench sample --data-dir $DATA --timeout 1800     # seconds per dataset
uv run agbench sample --data-dir $DATA --from-manifest    # re-download exact samples in manifests/
```

## Viewer

```bash
uv run agbench view --data-dir $DATA
```

## Benchmark

Rules: [docs/benchmarking.md](docs/benchmarking.md)

Inference servers (on the GPU machine):

```bash
uv tool install vllm                        # or set VLLM_BIN
export LLAMA_SERVER_BIN=/path/to/llama-server   # llama.cpp build >= b10353, for GGUF models
```

```bash
uv run agbench eval models
uv run agbench eval run --model Qwen/Qwen3.6-35B-A3B-FP8 --data-dir $DATA --judge
uv run agbench eval run --model Qwen/Qwen3.6-35B-A3B-FP8 --data-dir $DATA --task classification --format mcqa_within
uv run agbench eval run --model Qwen/Qwen3.6-35B-A3B-FP8 --data-dir $DATA --endpoint http://localhost:8000
uv run agbench eval judge --run runs/<run_dir>
uv run agbench eval report --runs runs
uv run agbench eval view --runs runs --profile reports/profile
```

Options:

```bash
--only '^AgroBench$'           # regex on dataset name
--limit-per-dataset 5
--no-thinking
--reasoning-effort low         # low | medium | high
--concurrency 16
--run-dir runs/<run_dir>       # resume
```

Compute profile:

```bash
scripts/profile_all_models.sh $DATA
uv run agbench eval profile --data-dir $DATA --models google/gemma-4-12B-it --per-task 10
uv run agbench eval view --runs reports/profile --profile reports/profile
```

Custom method:

```bash
uv add "agbench @ git+ssh://git@github.com/plant-ai-biophysics-lab/AGgentic-benchmarks"
```

```python
from agbench.eval import Benchmark, Method, EvalItem

class MyMethod(Method):
    name = "my-method"
    async def predict(self, item: EvalItem) -> str:
        ...  # item.images, item.prompt, item.choices, item.task, item.format, item.crop, item.ag_task

res = Benchmark("/path/to/data_dir", tasks=["classification"]).run(MyMethod(), out_dir="runs/my-method")
res.metrics
```

```bash
uv run agbench eval run --method mypkg.methods:MyMethod --data-dir $DATA
uv run agbench eval check-agtasks           # ag-task mapping gaps after a catalog sync
```

## Tests

```bash
uv run pytest
```

## Layout

```
configs/base_data.yaml                      sampling settings and per-dataset overrides
catalog/catalog.json                        normalized AgML catalog
catalog/CHANGES.md                          catalog changes per sync
manifests/base-data/<task>/<dataset>.json   sampled ids, labels, HF revision
src/agbench/eval/*.yaml                     prompts, model registry, ag tasks, segmentation class names
runs/<model>_<time>/                        items.jsonl, predictions.jsonl, metrics.json, crops.json
reports/profile/                            report.md, report.html, profile.json
$DATA/base-data/<task>/<dataset>/           images/, masks/, samples.jsonl, meta.json
```
