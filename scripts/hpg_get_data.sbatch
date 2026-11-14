#!/bin/bash
#SBATCH --job-name=fedtry-data
#SBATCH --account=gaitor_club
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Needs SYNAPSE_AUTH_TOKEN exported in the shell that runs sbatch.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/bin/activate
SKIP_MSD="${SKIP_MSD:-1}" bash get_data.sh data
ls data/fets2022 | head
