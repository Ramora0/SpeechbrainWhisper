#!/bin/bash
# BP4 ablation sweep: diagnose why BoundaryPredictor4 doesn't converge.
# Uses bp4_10x.yaml with CLI overrides.
set -euo pipefail

SLURM="slurms/train-v100.slurm"
YAML="hparams/bp4_10x.yaml"
S=58008  # seed

# --- 1: Forced all boundaries (no compression, sanity check) ---
# If this fails, the problem is mean pooling or something deeper.
sbatch --job-name=bp4_forced_all "$SLURM" "$YAML" \
  --boundary_mode=all --boundary_predictor_loss_weight=0 \
  --max_batch_length_train=500 \
  --experiment_name=bp4_forced_all --seed=$S

# --- 2: Forced alternating boundaries (~2x compression) ---
# Tests whether mean pooling works with actual compression.
sbatch --job-name=bp4_forced_alt "$SLURM" "$YAML" \
  --boundary_mode=alternating --boundary_predictor_loss_weight=0 \
  --experiment_name=bp4_forced_alt --seed=$S

# --- 3: Learned, prior=1.0 (learn to keep all boundaries) ---
# If this fails but #1 passes, the MLP can't learn even trivially.
sbatch --job-name=bp4_prior1 "$SLURM" "$YAML" \
  --boundary_predictor_prior=1.0 \
  --max_batch_length_train=500 \
  --experiment_name=bp4_learned_prior1 --seed=$S

# --- 4: Learned, prior=0.5 (learn 2x compression) ---
sbatch --job-name=bp4_prior05 "$SLURM" "$YAML" \
  --boundary_predictor_prior=0.5 \
  --experiment_name=bp4_learned_prior05 --seed=$S

# --- 5: Learned, original prior + CTC training loss ---
# CTC gives frame-level gradient signal to encoder, which helps MLP inputs.
sbatch --job-name=bp4_ctc03 "$SLURM" "$YAML" \
  --ctc_weight=0.3 \
  --experiment_name=bp4_learned_ctc03 --seed=$S

echo "All 5 BP4 ablation jobs submitted."
