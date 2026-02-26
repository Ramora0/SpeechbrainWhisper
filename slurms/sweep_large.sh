#!/bin/bash
# Sweep all large conformer and BP experiments on A100 with 12h time limit.
# Conformer-only configs use pretrain_raw.py (--raw flag).
# BP configs use pretrain.py (default).

set -euo pipefail

SLURM_SCRIPT="slurms/train-a100.slurm"
TIME="12:00:00"

# Large conformer baselines (no BP) — skip 2x per request
for config in 4x 8x 16x 32x; do
    echo "Submitting large conformer ${config}..."
    sbatch --time="${TIME}" "${SLURM_SCRIPT}" --raw "hparams/large/conformer_${config}.yaml"
done

# Large BP experiments
for config in 2x_2x 2x_4x 2x_8x 2x_16x 2x_p 2x_5-2; do
    echo "Submitting large BP ${config}..."
    sbatch --time="${TIME}" "${SLURM_SCRIPT}" "hparams/large/bp/${config}.yaml"
done

echo "All jobs submitted."
