#!/bin/bash
# Submit the whole pipeline on HiPerGator: download -> preprocess -> training grid.
#   export SYNAPSE_AUTH_TOKEN='...'      # only needed if the data is not downloaded yet
#   bash run_all.sh                       # pilot grid, 30 runs, 4 GPUs at a time
#   GRID=experiments/full.txt MAXPAR=8 bash run_all.sh
#   ARRAY=2-29 bash run_all.sh            # only these grid lines (0-based)
# When the training jobs finish:  python scripts/analyze.py
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
GRID="${GRID:-experiments/pilot.txt}"
MAXPAR="${MAXPAR:-4}"
N=$(grep -c . "$GRID")
ARRAY="${ARRAY:-0-$((N - 1))}"
dep=""

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

j=$(sbatch --parsable $dep --export=ALL,GRID="$GRID" --array="$ARRAY"%"$MAXPAR" scripts/hpg_train.sbatch)
echo "training array $j: tasks $ARRAY of $N in $GRID, $MAXPAR at a time"
echo "watch:   squeue -u \$USER"
echo "results: python scripts/analyze.py   (works on finished runs at any time)"
