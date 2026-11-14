#!/usr/bin/env bash
# Fetch the data for the split-point study. Run on the GPU machine.
#   bash get_data.sh /path/to/data
set -euo pipefail
OUT="${1:-./data}"
mkdir -p "$OUT/msd_task01" "$OUT/fets2022"

# 1) Public, no registration: Medical Segmentation Decathlon Task01 (BraTS 2016/17 subset)
#    484 labelled 4-modality cases, 7.6 GB, licence CC-BY-SA 4.0. No institution labels.
curl -L -C - -o "$OUT/msd_task01/Task01_BrainTumour.tar" \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xf "$OUT/msd_task01/Task01_BrainTumour.tar" -C "$OUT/msd_task01"

# 2) Gated: FeTS 2022 (= BraTS 2021 training set, 1251 cases, with institution partition).
#    Needs an approved data request (https://www.synapse.org/Synapse:syn54079892/wiki/626845)
#    and a Synapse personal access token in SYNAPSE_AUTH_TOKEN. Never commit the token.
#    syn29266807 = MICCAI_FeTS2022_TrainingData.zip, 13.4 GB, with labels.
#    (The 2.4 GB validation set, syn30107072, has no public labels and is not needed.)
#    The institution partition files are already in ./partitions/fets2022/.
if [ -n "${SYNAPSE_AUTH_TOKEN:-}" ]; then
  pip install -q synapseclient
  mkdir -p "$OUT/fets2022"
  python3 - "$OUT/fets2022" <<'PY'
import os, sys, synapseclient
syn = synapseclient.Synapse(silent=True)
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"], silent=True)
f = syn.get("syn29266807", downloadLocation=sys.argv[1])
print("saved", f.path)
PY
  unzip -q "$OUT/fets2022/MICCAI_FeTS2022_TrainingData.zip" -d "$OUT/fets2022"
else
  echo "SYNAPSE_AUTH_TOKEN not set: skipping FeTS 2022."
fi
echo "Done. Data in $OUT."
