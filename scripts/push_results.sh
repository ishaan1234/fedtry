#!/bin/bash
# Copy run results into results/ and push them to GitHub.
#
# Copies only small text files: config, logs, per-case scores. Never images,
# preprocessed data or checkpoints. Safe to run from several jobs at once
# (a lock serializes git) and safe to run by hand on a login node:
#   bash scripts/push_results.sh
set -uo pipefail
cd "$(dirname "$0")/.."
msg="${1:-results from $(hostname -s) $(date -u +%Y-%m-%d_%H:%M)}"

lock=.git/push_results.lock
for _ in $(seq 1 120); do
  mkdir "$lock" 2>/dev/null && break
  # Break a lock left behind by a killed job (older than 15 minutes).
  if [ -n "$(find "$lock" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then rmdir "$lock" 2>/dev/null; fi
  sleep 5
done
[ -d "$lock" ] || { echo "push_results: could not get lock" >&2; exit 1; }
trap 'rmdir "$lock" 2>/dev/null' EXIT

mkdir -p results/runs results/logs
for d in runs/*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  mkdir -p "results/runs/$n"
  for f in config.json summary.json test_cases.csv train_log.csv val_log.csv finetune_log.csv; do
    [ -f "$d$f" ] && cp "$d$f" "results/runs/$n/"
  done
done
for f in logs/*.out; do [ -f "$f" ] && cp "$f" results/logs/; done

git add results/runs results/logs
if git diff --cached --quiet; then echo "push_results: nothing new"; exit 0; fi
git -c user.name="${GIT_NAME:-hipergator}" -c user.email="${GIT_EMAIL:-hipergator@users.noreply.github.com}" \
  commit -q -m "$msg"
for i in 1 2 3 4 5; do
  if git pull -q --rebase origin main && git push -q origin HEAD:main; then
    echo "push_results: pushed ($msg)"; exit 0
  fi
  sleep $((5 * i + RANDOM % 10))
done
echo "push_results: push failed; results are committed locally" >&2
exit 1
