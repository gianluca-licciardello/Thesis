#!/usr/bin/env bash
# Run three AIDE settings and five PPB settings in each subject condition.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MDERNET_INPUTS:?Set MDERNET_INPUTS to the exported input bundle}"
for dataset in aide_full aide_clean aide_balanced; do
    python -u current/run_subsets.py --dataset "$dataset" --data-root "$MDERNET_INPUTS" "$@"
done
for split in independent mixed; do
    for dataset in ppb_full ppb_clean ppb_eeg ppb_epq ppb_cluster; do
        python -u current/run_subsets.py --dataset "$dataset" --split "$split" --data-root "$MDERNET_INPUTS" "$@"
    done
done
