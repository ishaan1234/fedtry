#!/bin/bash
#SBATCH --job-name=fedtry-smoke
#SBATCH --account=gaitor_club
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -m pytest -q tests
python scripts/gpu_smoke.py --patch 128 --batch 4 --steps 20
