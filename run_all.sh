#!/bin/bash
# Submit the whole pipeline on HiPerGator: download -> preprocess -> training grid.
#   export SYNAPSE_AUTH_TOKEN='...'      # only needed if the data is not downloaded yet
#   bash run_all.sh                       # pilot grid, 30 runs, 3 GPUs at a time
#   GRID=experiments/full.txt bash run_all.sh
#   ARRAY=2-29 bash run_all.sh            # only these grid lines (0-based)
#   GRID=experiments/round2.txt bash run_all.sh   # 141 runs: 5 folds + split+fine-tuning
#   AFTER=51723502 GRID=... bash run_all.sh       # start only after that job has ended
# Defaults keep this project at about 40% of the group quota:
#   MAXPAR=3 tasks at once x (CPUS=4, MEM=32gb, 1 GPU) = 12 CPUs, 96 GB, 3 GPUs.
# When the training jobs finish:  python scripts/analyze.py
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
GRID="${GRID:-experiments/pilot.txt}"
MAXPAR="${MAXPAR:-3}"
CPUS="${CPUS:-4}"
MEM="${MEM:-32gb}"
N=$(grep -c . "$GRID")
ARRAY="${ARRAY:-0-$((N - 1))}"
dep=""
[ -n "${AFTER:-}" ] && dep="--dependency=afterany:$AFTER"

if [ ! -f data/fets2022_prep/index.csv ]; then
  if [ -z "$(find data/fets2022 -name '*_seg.nii.gz' -print -quit 2>/dev/null)" ]; then
    if [ -z "${SYNAPSE_AUTH_TOKEN:-}" ]; then
      echo "FeTS data not found and SYNAPSE_AUTH_TOKEN is not set." >&2; exit 1
    fi
    j=$(sbatch --parsable scripts/hpg_get_data.sbatch)
    echo "download job  $j"; dep="--dependency=afterok:$j"
  fi
  j=$(sbatch --parsable $dep scripts/hpg_prepare.sbatch)
  echo "prepare job   $j"; dep="--dependency=afterok:$j"
fi

j=$(sbatch --parsable $dep --export=ALL,GRID="$GRID" --array="$ARRAY"%"$MAXPAR" \
  --cpus-per-task="$CPUS" --mem="$MEM" scripts/hpg_train.sbatch)
echo "training array $j: tasks $ARRAY of $N in $GRID, $MAXPAR at a time, $CPUS CPUs + $MEM each"
if [ -n "${AFTER:-}" ]; then echo "waits for job $AFTER to end"; fi
echo "watch:   squeue -u \$USER"
echo "results: python scripts/analyze.py   (works on finished runs at any time)"
