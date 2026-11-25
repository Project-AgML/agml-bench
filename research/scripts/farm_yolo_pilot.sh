#!/bin/bash
#SBATCH --job-name=agml-bench-yolo-pilot
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --account=jmearlesgrp
#SBATCH --partition=gpum
#SBATCH --qos=gpum-users-gpum-qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
# YOLO speed pilot: 15 shots, subset 0, five representative datasets, YOLO11s then YOLO11x.
# Per-dataset train/score times land in runs/yolo_pilot/<size>/15shot/s0/compute.json.
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$PATH
cd /group/jmearlesgrp/scratch/eranario/AGgentic-benchmarks
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
ONLY='^(date_grade_variety_classification|date_palm_leaf_disease_classification|tomato_ripeness_detection|cocoa_pod_disease_gh_image_classification_2|pine_wilt_segmentation)$'
uv sync --extra yolo -q
for S in s x; do
  date "+%F %T start $S"
  uv run --extra yolo agml-bench eval yolo --data-dir $DATA --shots 15 --subset 0 --size $S --only "$ONLY" --out runs/yolo_pilot/$S
  date "+%F %T end $S rc=$?"
done
