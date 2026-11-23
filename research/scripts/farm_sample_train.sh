#!/bin/bash
#SBATCH --job-name=agbench-sample-train
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=publicgrp
#SBATCH --partition=low
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
# usage: sbatch farm_sample_train.sh [extra agbench sample-train args]
# Few-shot training images for the YOLO baselines into $DATA/train-data (resumable: finished datasets are skipped,
# so a preempted/requeued job continues where it stopped). Manifests go to manifests/train-data/.
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$PATH
cd /group/jmearlesgrp/scratch/eranario/AGgentic-benchmarks
uv run agbench sample-train --data-dir $DATA --skip-failed "$@"
