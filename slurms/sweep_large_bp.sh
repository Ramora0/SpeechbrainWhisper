#!/bin/bash
# Sweep all large BP experiments on A100 with 12h time limit.

set -euo pipefail

SLURM_SCRIPT="slurms/train-a100.slurm"
TIME="12:00:00"

for config in 2x_2x 2x_4x 2x_8x 2x_16x 2xp 2x_5-2; do
    echo "Submitting large BP ${config}..."
    sbatch --time="${TIME}" "${SLURM_SCRIPT}" "hparams/large/bp/${config}.yaml"
done

echo "All jobs submitted."
