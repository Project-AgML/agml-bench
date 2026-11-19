# Paper experiments

Frozen paper version: tag `paper-v1` in plant-ai-biophysics-lab/AGgentic-benchmarks.

## Runs

```bash
DATA=/path/to/data_dir

DATA=$DATA research/scripts/run_plan.sh research/experiments/seeds.plan     # Experiment 1
DATA=$DATA research/scripts/run_plan.sh research/experiments/exp24.plan     # Experiments 2-4
for s in 0 1 2; do uv run agbench eval baseline --data-dir $DATA --kind chance --gen-seed $s --out runs/baselines/chance/s$s; done
```

YOLO11 baselines (farm, Slurm):

```bash
J=$(sbatch --parsable research/scripts/farm_sample_train.sh)
sbatch --dependency=afterok:$J --export=ALL,YOLO_CMD=yolo-shared,YOLO_SIZE=s,PER_GPU=2 research/scripts/farm_yolo.sh research/experiments/yolo_shared.plan
```

## Reproduce

```bash
bash research/paper/scripts/reproduce.sh                                    # stats, tables, figures, number audit, PDF
bash research/paper/scripts/watch.sh                                        # rebuild the PDF on edits
uv run python research/scripts/replay_paper.py --data-dir $DATA             # paper answers through agml-full-v1: scores unchanged
```
