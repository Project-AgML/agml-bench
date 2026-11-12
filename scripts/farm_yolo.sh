#!/bin/bash
#SBATCH --job-name=agbench-yolo
#SBATCH --output=farm_outputs/%A.out
#SBATCH --error=farm_outputs/%A.err
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@900
# usage (partition, GPUs and CPUs on the command line; several jobs can share one plan). YOLO_CMD=yolo-shared trains one
# model per task (experiments/yolo_shared.plan, results in runs/yolo11<size>-shared/); the default, yolo, one per dataset:
#   sbatch --account=geminigrp --partition=gpu-a100-h --gpus-per-node=1 --cpus-per-task=32 --mem=120G \
#          --export=ALL,YOLO_CMD=yolo-shared,YOLO_SIZE=s,PER_GPU=2 scripts/farm_yolo.sh experiments/yolo_shared.plan
#   sbatch --account=geminigrp --partition=gpu-a100-h --gpus-per-node=1 --cpus-per-task=32 --mem=120G \
#          --export=ALL,YOLO_SIZE=s,PER_GPU=8 scripts/farm_yolo.sh experiments/yolo.plan
#   sbatch --account=jmearlesgrp --partition=gpum --qos=gpum-users-gpum-qos --gpus-per-node=2 --cpus-per-task=20 --mem=64G \
#          --export=ALL,YOLO_SIZE=s,PER_GPU=2 scripts/farm_yolo.sh experiments/yolo.plan
# Every worker (PER_GPU per GPU) works through the unfinished plan lines (<shots> <subset>) in plan order, so the lines
# finish one after another (put the most important first). agbench eval yolo claims each dataset (yolo-shared: each task) with a lock file, so any
# number of workers in any number of jobs share a line without duplicating work, and a dataset claimed by a job that died
# is picked up again. 15 min before the time limit the job resubmits
# itself with the same settings if lines remain. Progress in <plan>.log; results in runs/yolo11<size>/.
YOLO_SIZE=${YOLO_SIZE:-s}
YOLO_CMD=${YOLO_CMD:-yolo}
OUT=runs/yolo11$YOLO_SIZE; [ "$YOLO_CMD" = yolo-shared ] && OUT=runs/yolo11$YOLO_SIZE-shared
PER_GPU=${PER_GPU:-2}
export DATA=/group/jmearlesgrp/data/AGgentic-benchmarks
export PATH=$HOME/.local/bin:$PATH
cd /group/jmearlesgrp/scratch/eranario/AGgentic-benchmarks
PLAN=$1; DONE=$PLAN.done; LOG=$PLAN.log; touch $DONE
log() { echo "$(date '+%F %T') job $SLURM_JOB_ID $*" >> $LOG; }
todo() { grep -v '^\s*#' $PLAN | grep -v '^\s*$' | grep -v -x -F -f $DONE; }
resubmit() {
  log "time limit near: resubmitting"
  sbatch --account=$SLURM_JOB_ACCOUNT --partition=$SLURM_JOB_PARTITION --qos=$SLURM_JOB_QOS \
         --gpus-per-node=$SLURM_GPUS_PER_NODE --cpus-per-task=$SLURM_CPUS_PER_TASK --mem=${SLURM_MEM_PER_NODE}M \
         --export=ALL,YOLO_CMD=$YOLO_CMD,YOLO_SIZE=$YOLO_SIZE,PER_GPU=$PER_GPU scripts/farm_yolo.sh $PLAN >> $LOG
  kill $(jobs -p) 2>/dev/null; exit 0
}
trap resubmit USR1

( flock 7; uv sync --extra yolo -q ) 7>$HOME/.cache/agbench-uv-sync.lock
W=$HOME/.cache/agbench/yolo; mkdir -p $W
( flock 8; cd $W && uv run --project /group/jmearlesgrp/scratch/eranario/AGgentic-benchmarks --extra yolo python -c \
  "from ultralytics import YOLO; [YOLO(f'yolo11${YOLO_SIZE}{t}.pt') for t in ('-cls', '', '-seg')]" ) 8>$W/.download.lock

worker() {  # works through the unfinished plan lines in plan order; datasets are claimed inside agbench eval yolo,
  local gpu=$1 idx=$2 lines n j l rc progress   # so all workers share the first unfinished line before moving on
  while true; do
    mapfile -t lines < <(todo); n=${#lines[@]}
    [ $n -eq 0 ] && break
    progress=0
    for ((j = 0; j < n; j++)); do
      l=${lines[$j]}
      grep -qxF "$l" $DONE && continue
      read -r SHOTS SUBSET <<< "$l"
      CUDA_VISIBLE_DEVICES=$gpu uv run --extra yolo agbench eval $YOLO_CMD --data-dir $DATA --shots $SHOTS --subset $SUBSET \
        --size $YOLO_SIZE --out $OUT >> farm_outputs/${OUT#runs/}_${SHOTS}shot_s${SUBSET}.log 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then
        ( flock 9; grep -qxF "$l" $DONE || echo "$l" >> $DONE ) 9>$PLAN.lock; log "gpu $gpu worker $idx: line complete: $l"; progress=1
      elif [ $rc -ne 3 ]; then log "gpu $gpu worker $idx: line $l rc=$rc"; fi
    done
    [ $progress -eq 0 ] && sleep 120  # everything left is being trained by other workers
  done
}
GPUS=(${CUDA_VISIBLE_DEVICES//,/ })  # the GPUs Slurm allocated to this job
log "start: $SLURM_JOB_PARTITION GPUs ${GPUS[*]} x $PER_GPU workers, $YOLO_CMD yolo11$YOLO_SIZE"
w=0; for g in "${GPUS[@]}"; do for k in $(seq 1 $PER_GPU); do worker $g $w & w=$((w + 1)); sleep 5; done; done
wait
if [ -n "$(todo)" ]; then log "finished its work; lines still left or claimed elsewhere: $(todo | tr '\n' ';')"; else log "all done"; fi
