#!/bin/bash
#SBATCH --job-name=fedtry-train
#SBATCH --account=gaitor_club
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%A_%a.out
# One array task = one line of the grid file (default experiments/pilot.txt).
# Finished runs are skipped, interrupted runs resume, and a run already being
# trained by another job is left alone, so resubmitting is safe.
# 4 CPUs and 32 GB per task: 3 tasks use about 40% of the group limit
# (32 CPUs, 250 GB, 8 GPUs). Loader workers = CPUs - 1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Push results to GitHub every PUSH_EVERY seconds while training, and once at the end
# (success or failure), so progress and errors are visible remotely.
PUSH_EVERY="${PUSH_EVERY:-7200}"
( while sleep "$PUSH_EVERY"; do bash scripts/push_results.sh "progress: array task $SLURM_ARRAY_TASK_ID" || true; done ) &
PUSHER=$!
trap 'kill $PUSHER 2>/dev/null; bash scripts/push_results.sh "array task $SLURM_ARRAY_TASK_ID ended" || true' EXIT
source env/bin/activate
GRID="${GRID:-experiments/pilot.txt}"
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GRID")
echo "task $SLURM_ARRAY_TASK_ID: $ARGS"
python -m pflseg.train --data_dir data/fets2022_prep \
  --partition partitions/fets2022/partitioning_1.csv --workers $(( ${SLURM_CPUS_PER_TASK:-4} - 1 )) --out runs $ARGS
