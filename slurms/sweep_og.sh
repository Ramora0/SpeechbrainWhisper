#!/bin/bash
set -euo pipefail

sbatch --job-name=og_base --time=24:00:00 slurms/train-v100.slurm --raw hparams/large/og/baseline.yaml && \
sbatch --job-name=og_dyn2x --time=24:00:00 slurms/train-v100.slurm --raw hparams/large/og/dynamic_batch_2xlr.yaml && \
sbatch --job-name=og_30ep --time=24:00:00 slurms/train-v100.slurm --raw hparams/large/og/30_epochs.yaml

echo "All og jobs submitted."
