#!/bin/bash
# Sweep all large BP experiments on A100 with 12h time limit.
# Deletes old checkpoints for each run before submitting.

set -euo pipefail

SLURM_SCRIPT="slurms/train-a100.slurm"
TIME="12:00:00"
RESULTS_DIR="/fs/scratch/PAS2836/lees_stuff/librispeechbrain/results"

for config in 2x_2x 2x_4x 2x_8x 2x_16x 2xp 2x_5-2; do
    yaml="hparams/large/bp/${config}.yaml"
    exp_name=$(grep '^experiment_name:' "$yaml" | awk '{print $2}')
    seed=$(grep '^seed:' "$yaml" | awk '{print $2}')
    ckpt_dir="${RESULTS_DIR}/${exp_name}/${seed}"

    if [ -d "$ckpt_dir" ]; then
        echo "Deleting old checkpoints: ${ckpt_dir}"
        rm -rf "$ckpt_dir"
    fi

    echo "Submitting large BP ${config}..."
    sbatch --time="${TIME}" "${SLURM_SCRIPT}" "$yaml"
done

echo "All jobs submitted."
