#!/bin/bash
# Ablation round 2: CTC fix, consistent batch/LR, compression schedules.
# All experiments use mean pooling, pretrain_test.py, bp_ablation2.yaml.
set -euo pipefail

SLURM="slurms/train-v100.slurm"
YAML="hparams/ablation/bp_ablation2.yaml"
S=58008  # seed

# --- 1-2: CTC fix + convergence re-test (learned cosine & MLP) ---
sbatch --job-name=abl2_learned_cosine  "$SLURM" --test "$YAML" \
  --boundary_mode=learned --experiment_name=abl2_learned_cosine_ctc --seed=$S

sbatch --job-name=abl2_learned_mlp     "$SLURM" --test "$YAML" \
  --boundary_mode=mlp     --experiment_name=abl2_learned_mlp_ctc     --seed=$S

# --- 3: CTC decode comparison (both CTC train + CTC decode) ---
sbatch --job-name=abl2_ctc_both        "$SLURM" --test "$YAML" \
  --boundary_mode=learned --ctc_weight_decode=0.4 \
  --experiment_name=abl2_learned_cosine_ctc_both --seed=$S

# --- 4: Fixed boundary baseline (sanity check) ---
sbatch --job-name=abl2_fixed           "$SLURM" --test "$YAML" \
  --boundary_mode=fixed_width --experiment_name=abl2_fixed_width_ctc --seed=$S

# --- 5-6: 2x compression crash investigation ---
sbatch --job-name=abl2_learned_2x      "$SLURM" --test "$YAML" \
  --boundary_mode=learned     --boundary_predictor_prior=0.5 \
  --experiment_name=abl2_learned_2x --seed=$S

sbatch --job-name=abl2_fixed_2x        "$SLURM" --test "$YAML" \
  --boundary_mode=fixed_width --fixed_width=2 \
  --experiment_name=abl2_fixed_2x --seed=$S

# --- 7-8: Linear 1x -> 5.2x schedule (70ep and 30ep) ---
sbatch --job-name=abl2_lin_70          "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=linear \
  --experiment_name=abl2_sched_linear_1to5x_70ep --seed=$S

sbatch --job-name=abl2_lin_30          "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=linear \
  --number_of_epochs=30 \
  --experiment_name=abl2_sched_linear_1to5x_30ep --seed=$S

# --- 9-10: Hold 1x for 5ep then linear to 5.2x (70ep and 30ep) ---
sbatch --job-name=abl2_hold5_70        "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=hold_then_linear \
  --compression_hold_epochs=5 \
  --experiment_name=abl2_sched_hold5_linear_70ep --seed=$S

sbatch --job-name=abl2_hold5_30        "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=hold_then_linear \
  --compression_hold_epochs=5 --number_of_epochs=30 \
  --experiment_name=abl2_sched_hold5_linear_30ep --seed=$S

# --- 11: Linear 1x -> 2x over 70ep ---
sbatch --job-name=abl2_lin_1to2x       "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=linear \
  --boundary_predictor_prior=0.5 \
  --experiment_name=abl2_sched_linear_1to2x --seed=$S

# --- 12: Ramp 1x -> 2x over 10ep then hold at 2x ---
sbatch --job-name=abl2_hold_2x         "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=hold_at_target \
  --boundary_predictor_prior=0.5 --compression_ramp_epochs=10 \
  --experiment_name=abl2_sched_hold_at_2x --seed=$S

# --- 13-14: Step schedule (70ep and 30ep) ---
sbatch --job-name=abl2_step_70         "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=step \
  --compression_steps=0:1.0,16:2.0,36:3.5,51:5.2 \
  --experiment_name=abl2_sched_step_70ep --seed=$S

sbatch --job-name=abl2_step_30         "$SLURM" --test "$YAML" \
  --boundary_mode=learned --compression_schedule_type=step \
  --compression_steps=0:1.0,7:2.0,15:3.5,22:5.2 \
  --number_of_epochs=30 \
  --experiment_name=abl2_sched_step_30ep --seed=$S

echo "All 14 ablation round 2 jobs submitted."
