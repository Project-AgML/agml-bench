# Building benchmark versions

## Catalog

```bash
uv run agml-bench catalog sync                 # fetch latest AgML catalog, print what changed
uv run agml-bench catalog diff                 # show upstream changes without saving
uv run agml-bench catalog list                 # eligible datasets
uv run agml-bench catalog list --all           # include excluded datasets with reason
uv run agml-bench catalog list --task image_text_to_text
uv run agml-bench eval check-agtasks           # ag-task mapping gaps after a sync
```

## Sample

Rules: [subsampling.md](subsampling.md)

```bash
DATA=/path/to/data_dir

uv run agml-bench status --data-dir $DATA
uv run agml-bench sample --data-dir $DATA
uv run agml-bench sample --data-dir $DATA --task classification
uv run agml-bench sample --data-dir $DATA --only '^AgroBench$'
uv run agml-bench sample --data-dir $DATA --refresh          # resample datasets already done
uv run agml-bench sample --data-dir $DATA --skip-failed      # do not retry failed datasets
uv run agml-bench sample --data-dir $DATA --timeout 1800     # seconds per dataset
uv run agml-bench sample --data-dir $DATA --from-manifest    # re-download exact samples in manifests/
uv run agml-bench view --data-dir $DATA
```

YOLO training images (`train_sampling` in [configs/base_data.yaml](../configs/base_data.yaml)):

```bash
uv run agml-bench sample-train --data-dir $DATA
```

## Freeze

```bash
uv run agml-bench eval run --model <model> --data-dir $DATA --benchmark none
uv run agml-bench bench freeze agml-full-v2 --items runs/<run_dir>/items.jsonl --description "..."
uv run agml-bench bench show agml-full-v2
```

## Source repos

```bash
uv run agml-bench bench repin agml-full-v1 --dataset CDDM --reference /path/to/frozen_copy           # report only
uv run agml-bench bench repin agml-full-v1 --dataset CDDM --reference /path/to/frozen_copy --write   # re-uploaded repo
```

## Ablations

```bash
--order-seed 1 --nota-position random   # MCQA: shuffle option order, NOTA/IDK anywhere
--nota-rate 0.5                         # MCQA: share of items whose true class is removed
--no-idk                                # MCQA: drop "I don't know"
--seed 1                                # item seed: which images, distractors and NOTA items
```

## Compute profile

```bash
uv run agml-bench eval models
uv run agml-bench eval profile --data-dir $DATA --models google/gemma-4-12B-it --per-task 5
uv run agml-bench eval profile --data-dir $DATA --models all
uv run agml-bench eval profile --data-dir $DATA --models all --redo
uv run agml-bench eval view --runs reports/profile --profile reports/profile
```

## llama.cpp (GGUF models)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j 8 --target llama-server
export LLAMA_SERVER_BIN=$PWD/build/bin/llama-server
```

## Layout

```
src/agml_bench/benchmarks/<name>/              frozen benchmark: bundle.yaml, items.jsonl.gz, manifests.json.gz, config.yaml
configs/base_data.yaml                      sampling settings and per-dataset overrides
catalog/catalog.json                        normalized AgML catalog
catalog/CHANGES.md                          catalog changes per sync
manifests/base-data/<task>/<dataset>.json   sampled ids, labels, HF revision
src/agml_bench/eval/*.yaml                     prompts, model registry, ag tasks, segmentation class names
runs/<model>_<time>/                        items.jsonl, predictions.jsonl, metrics.json, run.json
$DATA/base-data/<task>/<dataset>/           images/, masks/, samples.jsonl, meta.json
```
