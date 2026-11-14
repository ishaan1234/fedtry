#!/bin/bash
#SBATCH --job-name=fedtry-prep
#SBATCH --account=gaitor_club
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Push this job's log to GitHub when it ends, whether it succeeded or failed.
trap 'bash scripts/push_results.sh "$SLURM_JOB_NAME $SLURM_JOB_ID ended" || true' EXIT
source env/bin/activate
python scripts/prepare_fets.py --src data/fets2022 --out data/fets2022_prep --workers 16
