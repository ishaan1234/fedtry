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
#    Register and submit the data request first:
#      https://www.synapse.org/Synapse:syn54079892/wiki/626845
#    Then, with a Synapse personal access token and the Synapse ID of the training
#    archive shown on the challenge Files page after approval:
#      pip install synapseclient
#      synapse login -p "$SYNAPSE_AUTH_TOKEN"
#      synapse get -r <SYNAPSE_ID_OF_TRAINING_DATA> --downloadLocation "$OUT/fets2022"
#    The institution partition files are already in ./partitions/fets2022/.
echo "MSD Task01 ready in $OUT/msd_task01. FeTS 2022 needs your Synapse approval; see comments."
