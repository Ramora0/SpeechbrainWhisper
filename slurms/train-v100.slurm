#!/bin/bash
#SBATCH --job-name=train_bp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --constraint=48core
#SBATCH --time=12:00:00
#SBATCH --account=PAS2836
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  sbatch [--time=H:00:00] train-v100.slurm [--raw] <compression_name>"
  echo "  sbatch [--time=H:00:00] train-v100.slurm [--raw] /full/path/to/hparams.yaml"
  echo "Examples:"
  echo "  sbatch --time=8:00:00 slurms/train-v100.slurm 2x_5-2x"
  echo "  sbatch slurms/train-v100.slurm --raw hparams/large/og.yaml"
  exit 1
fi

# Check for --raw / --test flag
TRAIN_SCRIPT="pretrain.py"
if [[ "$1" == "--raw" ]]; then
  TRAIN_SCRIPT="pretrain_raw.py"
  shift
elif [[ "$1" == "--test" ]]; then
  TRAIN_SCRIPT="pretrain_test.py"
  shift
fi

ARG1="$1"
shift

# If ARG1 is an existing file, use it directly; otherwise treat it as a compression name.
if [[ -f "$ARG1" ]]; then
  HPARAMS_PATH="$ARG1"
else
  HPARAMS_PATH="hparams/bp/${ARG1}.yaml"
fi

if [[ ! -f "$HPARAMS_PATH" ]]; then
  echo "ERROR: hparams file not found: $HPARAMS_PATH"
  exit 2
fi

echo "JobID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Script: ${TRAIN_SCRIPT}"
echo "Hparams: ${HPARAMS_PATH}"

module purge
module load cuda/12.4.1  # Use 'module spider cuda' to see available versions

cd "${SLURM_SUBMIT_DIR}"

mkdir -p "$TMPDIR/LibriSpeech" "$TMPDIR/data_manifests"

rsync -a --info=progress2 /fs/scratch/PAS2836/lees_stuff/librispeechbrain/LibriSpeech/ "$TMPDIR/LibriSpeech/" && \
rsync -a --info=progress2 /fs/scratch/PAS2836/lees_stuff/librispeechbrain/data_manifests/ "$TMPDIR/data_manifests/"

source .v100/bin/activate
python "${TRAIN_SCRIPT}" "${HPARAMS_PATH}" "$@"
