# agml-benchmarks

Benchmark vision-language models on AgML agricultural datasets: classification, detection, segmentation and VQA.

## Install

```bash
git clone git@github.com:Project-AgML/agml-benchmarks.git && cd agml-benchmarks
uv sync
hf auth login
uv tool install vllm          # to serve models with --model
```

## Quick start

```bash
DATA=/path/to/data_dir

uv run agbench data pull agml-full-v1 --data-dir $DATA
uv run agbench eval run --model my-org/My-VLM --data-dir $DATA --judge
uv run agbench eval report --runs runs
```

## Run your model

Hugging Face repo or local checkpoint (served with vLLM):

```bash
uv run agbench eval run --model my-org/My-VLM --data-dir $DATA --judge
uv run agbench eval run --model /path/to/checkpoint --data-dir $DATA --judge
```

Already served (any OpenAI-compatible endpoint):

```bash
uv run agbench eval run --model my-org/My-VLM --endpoint http://localhost:8000 --data-dir $DATA --judge
```

Model settings:

```bash
uv run agbench eval run --model my-org/My-VLM --data-dir $DATA --models-file my_models.yaml --judge
```

```yaml
# my_models.yaml (fields as in src/agbench/eval/models.yaml)
my-org/My-VLM:
  serve_args: [--max-model-len, "32768"]
  sampling: {temperature: 0.7, top_p: 0.95}
  thinking: {chat_template_kwargs: {enable_thinking: true}}
  no_thinking: {chat_template_kwargs: {enable_thinking: false}}
  box_order: xyxy                       # box_2d as [x_min, y_min, x_max, y_max]; default yxyx
  bbox_key_format: {bbox_2d: xyxy_px}   # other box keys: xyxy_1000 | xyxy_px
```

Your own method:

```bash
uv run agbench eval run --method mypkg.methods:MyMethod --data-dir $DATA
```

```python
from agbench.eval import Benchmark, EvalItem, Method

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
```

## Results

```bash
uv run agbench eval judge --run runs/<run_dir>
uv run agbench eval report --runs runs
uv run agbench eval report --runs runs --csv reports/scores     # scores.csv, ag_tasks.csv, datasets.csv
uv run agbench eval stats --runs runs --out reports/stats
uv run agbench eval view --runs runs                            # http://localhost:8502
```

## Benchmarks

```bash
uv run agbench bench list
uv run agbench bench show agml-full-v1
uv run agbench data pull agml-full-v1 --data-dir $DATA --task classification
uv run agbench data status agml-full-v1 --data-dir $DATA
uv run agbench data verify agml-full-v1 --data-dir $DATA
```

## Baselines

```bash
uv run agbench eval baseline --kind chance --data-dir $DATA --out runs/baselines/chance
```

```bash
uv sync --extra yolo
uv run agbench sample-train --data-dir $DATA --from-manifest
uv run --extra yolo agbench eval yolo-shared --data-dir $DATA --shots 10 --subset 0
```

## More

- New benchmark versions, catalog, sampling, profiling: [docs/building.md](docs/building.md)
- Paper experiments and reproduction: [research/README.md](research/README.md)
- Evaluation rules: [docs/benchmarking.md](docs/benchmarking.md)

## Tests

```bash
uv run pytest
```
