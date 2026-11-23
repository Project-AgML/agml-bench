#!/bin/bash
# Rebuild every number, table and figure in the paper from run outputs, then the PDF (paper Appendix D).
#
# usage (from the repo root):
#   bash research/paper/scripts/reproduce.sh            # from the paper's run outputs (default paths below); CPU only
#   AGBENCH_SEEDS=runs/exp/seeds AGBENCH_EXP24=runs/exp24 bash research/paper/scripts/reproduce.sh
#                                              # from your own re-runs of experiments/*.plan
#
# Inputs:
#   AGBENCH_SEEDS  Experiment 1 run dirs, one subdir per seed (s0, s1, s2)    default runs/rejudge_v4/exp/seeds
#   AGBENCH_EXP24  Experiments 2-4 run dirs (order/, nota/, noidk/) and nothing else   default runs/runpod_exp/runs
#   AGBENCH_STATS  where agbench eval stats writes seeds/ and exp24/          default research/reports/stats
set -euo pipefail
cd "$(dirname "$0")/../../.."
export AGBENCH_SEEDS=${AGBENCH_SEEDS:-runs/rejudge_v4/exp/seeds}
export AGBENCH_EXP24=${AGBENCH_EXP24:-runs/runpod_exp/runs}
export AGBENCH_STATS=${AGBENCH_STATS:-research/reports/stats}

# 1. Statistics (bootstrap CIs, paired tests, NOTA/IDK behaviour, seed and option-order stability).
uv run agbench eval stats --runs "$AGBENCH_SEEDS" --out "$AGBENCH_STATS/seeds" \
  --title "Experiment 1: generation seeds (judge v4)"
uv run agbench eval stats --runs "$AGBENCH_EXP24" --runs "$AGBENCH_SEEDS/s0" \
  --out "$AGBENCH_STATS/exp24" --title "Experiments 2-4: option order, NOTA rate, IDK removed"

# 2. Per-run scores per task, agricultural task and dataset (CSV; not used by the paper scripts).
uv run agbench eval report --runs "$AGBENCH_SEEDS" --csv research/reports/scores/seeds
uv run agbench eval report --runs "$AGBENCH_EXP24" --csv research/reports/scores/exp24

# 3. Tables, figures and research/paper/numbers.json (every number the text quotes from the runs).
uv run --with matplotlib python research/paper/scripts/make_assets.py      # tables/*.tex, Figures 4-5, numbers.json
uv run --with matplotlib python research/paper/scripts/make_behaviour.py   # Figure 6
uv run --with matplotlib python research/paper/scripts/make_position.py    # Figure 7
uv run --with numpy python research/paper/scripts/make_drivers.py          # Table C1
uv run --with matplotlib --with pillow --with numpy --with pyyaml python research/paper/scripts/make_judge.py  # Figures 3, A1
uv run --with matplotlib --with pillow --with numpy python research/paper/scripts/make_overview.py             # Figure 1
uv run --with matplotlib --with pillow --with numpy python research/paper/scripts/make_prompts.py              # Figure 2

# 4. Audit: every number in the text vs the computed values (research/paper/build/number_check.txt).
uv run python research/paper/scripts/check_numbers.py | head -8

# 5. PDF.
cd research/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex && cp build/main.pdf main.pdf
