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
$DATA/base-data/<task>/<dataset>/           images/, masks/, samples.jsonl, meta.json
```
