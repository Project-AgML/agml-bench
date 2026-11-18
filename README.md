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

Setup:

```bash
git clone git@github.com:plant-ai-biophysics-lab/AGgentic-benchmarks.git && cd AGgentic-benchmarks
uv sync
hf auth login
uv tool install vllm
vllm download-kernels
```

llama.cpp (Nemotron GGUF):

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j 8 --target llama-server
export LLAMA_SERVER_BIN=$PWD/build/bin/llama-server
```

```bash
uv run agbench eval models
uv run agbench eval run --model Qwen/Qwen3.6-35B-A3B-FP8 --data-dir $DATA --judge
uv run agbench eval run --model Qwen/Qwen3.6-35B-A3B-FP8 --data-dir $DATA --task classification --format mcqa_within
uv run agbench eval run --model Qwen/Qwen3.6-35B-A3B-FP8 --data-dir $DATA --endpoint http://localhost:8000
uv run agbench eval judge --run runs/<run_dir>
uv run agbench eval judge --run runs/<run_dir> --redo   # re-judge everything (after editing the judge prompts)
uv run agbench eval report --runs runs
uv run agbench eval report --runs runs/full --csv reports/scores/full_low   # scores.csv, ag_tasks.csv, datasets.csv
uv run agbench eval stats --runs runs/exp/seeds --out reports/stats/seeds
uv run agbench eval view --runs runs --profile reports/profile
```

Options:

```bash
--only '^AgroBench$'           # regex on dataset name
--limit-per-dataset 5
--no-thinking
--reasoning-effort medium      # low (default) | medium | high
--concurrency 16
--gen-seed 1                   # sampling seed (replicate runs)
--order-seed 1 --nota-position random   # MCQA: shuffle option order, NOTA/IDK anywhere
--nota-rate 0.5                # MCQA: share of items whose true class is removed
--no-idk                       # MCQA: drop "I don't know"
--run-dir runs/<run_dir>       # resume (with --judge: judge whatever is left)
```

Compute profile:

```bash
uv run agbench eval run --model openbmb/MiniCPM-V-4.7-35B-A3B-FP8 --data-dir $DATA --limit-per-dataset 2 --only '^(bean_disease_uganda|almond_bloom_2023|apple_flower_segmentation|AgriBench)$'
uv run agbench eval profile --data-dir $DATA --models google/gemma-4-12B-it --per-task 5
uv run agbench eval profile --data-dir $DATA --models all
uv run agbench eval profile --data-dir $DATA --models all --redo   # re-profile models already done
research/scripts/profile_all_models.sh $DATA                                 # same, deleting each model's weights after
```

Resume:

```bash
uv run agbench eval run --model Qwen/Qwen3.6-35B-A3B-FP8 --data-dir $DATA --judge --run-dir runs/<run_dir>
uv run agbench eval judge --run runs/<run_dir>
uv run agbench eval judge --run runs/<run_dir> --redo   # re-judge everything (after editing the judge prompts)
uv run agbench eval profile --data-dir $DATA --models all   # keeps models already in reports/profile/profile.json
```

View results (http://localhost:8502):

```bash
uv run agbench eval view --runs reports/profile --profile reports/profile
uv run agbench eval view --runs reports/profile --profile reports/profile --data-dir $DATA   # images under another base-data/ copy
```

Your own model (any HF repo or local path; vLLM by default):

```bash
uv run agbench eval run --model my-org/My-VLM --data-dir $DATA
uv run agbench eval run --model /path/to/checkpoint --data-dir $DATA
uv run agbench eval run --model my-org/My-VLM --data-dir $DATA --models-file my_models.yaml
uv run agbench eval profile --models my-org/My-VLM --data-dir $DATA --models-file my_models.yaml
uv run agbench eval run --model my-org/My-VLM --data-dir $DATA --endpoint http://localhost:8000
```

```yaml
# my_models.yaml (same fields as src/agbench/eval/models.yaml)
my-org/My-VLM:
  serve_args: [--reasoning-parser, qwen3, --quantization, awq]
  sampling: {temperature: 0.7, top_p: 0.95}
  thinking: {chat_template_kwargs: {enable_thinking: true}}
  no_thinking: {chat_template_kwargs: {enable_thinking: false}}
  box_order: xyxy                       # box_2d as [x_min, y_min, x_max, y_max]; default yxyx
  bbox_key_format: {bbox_2d: xyxy_px}   # other box keys: xyxy_1000 | xyxy_px
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
res = Benchmark("/path/to/data_dir", tasks=["detection"]).run(MyMethod(), out_dir="runs/my-method-det", key_formats={"box_2d": "xyxy_1000"})
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

## Naive baselines

```bash
uv run agbench eval baseline --data-dir $DATA --kind prior --out runs/baselines/prior
for s in 0 1 2; do uv run agbench eval baseline --data-dir $DATA --kind chance --gen-seed $s --out runs/baselines/chance/s$s; done
```

## Few-shot YOLO11 baselines

Training images (`train_sampling` in [configs/base_data.yaml](configs/base_data.yaml); never an evaluated sample):

```bash
uv run agbench sample-train --data-dir $DATA
uv run agbench sample-train --data-dir $DATA --from-manifest     # re-download exact samples in manifests/train-data/
```

```bash
uv sync --extra yolo
uv run --extra yolo agbench eval yolo --data-dir $DATA --shots 5 --subset 0 --out runs/yolo
uv run --extra yolo agbench eval yolo --data-dir $DATA --shots 5 --subset 0 --size n --epochs 2 --only '^bean_disease_uganda$'
```

Farm (Slurm):

```bash
J=$(sbatch --parsable research/scripts/farm_sample_train.sh)
sbatch --dependency=afterok:$J research/scripts/farm_yolo.sh research/experiments/yolo.plan
```

## Paper experiments

```bash
DATA=$DATA research/scripts/run_plan.sh research/experiments/seeds.plan     # Experiment 1
DATA=$DATA research/scripts/run_plan.sh research/experiments/exp24.plan     # Experiments 2-4
bash research/paper/scripts/reproduce.sh                            # stats, tables, figures, number audit, PDF
```
