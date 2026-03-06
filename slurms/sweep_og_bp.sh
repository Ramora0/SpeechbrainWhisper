#!/bin/bash
set -euo pipefail

sbatch --job-name=bp_2x_2x --time=24:00:00 slurms/train-v100.slurm hparams/large/bp/2x_2x.yaml

echo "BP 2x_2x job submitted."
