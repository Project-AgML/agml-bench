#!/bin/bash
# Run every line of an experiment plan (<out dir> <model> [agbench eval run args]) in order, one model at a time.
# Resumable: finished lines are recorded in <plan>.done; an interrupted run resumes in its run dir.
# usage: DATA=/path/to/base-data scripts/run_plan.sh experiments/seeds.plan
set -euo pipefail
PLAN=$1; DONE=$PLAN.done; touch "$DONE"
: "${DATA:?set DATA to the base-data dir (agbench sample --data-dir \$DATA)}"
grep -v '^\s*#' "$PLAN" | grep -v '^\s*$' | while read -r OUT MODEL ARGS; do
  LINE="$OUT $MODEL $ARGS"
  grep -qxF "$LINE" "$DONE" && continue
  DIR=$(ls -td "$OUT/${MODEL//\//__}_"* 2>/dev/null | head -1 || true)
  uv run agbench eval run --model "$MODEL" --data-dir "$DATA" --out "$OUT" $ARGS ${DIR:+--run-dir $DIR} </dev/null
  echo "$LINE" >> "$DONE"
done
