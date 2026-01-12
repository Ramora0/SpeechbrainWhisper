#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=slurms/test_%j.out
#SBATCH --error=slurms/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# Change to the project directory
cd /research/SpeechbrainWhisper

# Activate virtual environment
source .h100/bin/activate

# Run the training script with test configuration
python pretrain.py hparams/test.yaml
