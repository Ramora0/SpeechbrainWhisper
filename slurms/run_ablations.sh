#!/bin/bash
# Sweep all BoundaryPredictor ablation experiments on V100.
set -euo pipefail

SLURM="slurms/train-v100.slurm"
YAML="hparams/ablation/bp_ablation.yaml"
S=58008  # seed

# --- Grid ablations ---
# Control
sbatch --job-name=abl_learned_attn       "$SLURM" "$YAML" --boundary_mode=learned     --pooling_mode=attention          --experiment_name=ablation_learned_attention          --seed=$S
sbatch --job-name=abl_learned_mean       "$SLURM" "$YAML" --boundary_mode=learned     --pooling_mode=mean               --experiment_name=ablation_learned_mean               --seed=$S
sbatch --job-name=abl_learned_attn_nov   "$SLURM" "$YAML" --boundary_mode=learned     --pooling_mode=attention_no_value --experiment_name=ablation_learned_attention_no_value --seed=$S

# Fixed width
sbatch --job-name=abl_fixed_attn         "$SLURM" "$YAML" --boundary_mode=fixed_width --pooling_mode=attention          --experiment_name=ablation_fixed_width_attention          --seed=$S
sbatch --job-name=abl_fixed_mean         "$SLURM" "$YAML" --boundary_mode=fixed_width --pooling_mode=mean               --experiment_name=ablation_fixed_width_mean               --seed=$S
sbatch --job-name=abl_fixed_attn_nov     "$SLURM" "$YAML" --boundary_mode=fixed_width --pooling_mode=attention_no_value --experiment_name=ablation_fixed_width_attention_no_value --seed=$S

# All (no BP compression — reduce batch to avoid OOM)
sbatch --job-name=abl_all_mean           "$SLURM" "$YAML" --boundary_mode=all         --pooling_mode=mean               --experiment_name=ablation_all_mean               --seed=$S --max_batch_length_train=500
sbatch --job-name=abl_all_attn           "$SLURM" "$YAML" --boundary_mode=all         --pooling_mode=attention          --experiment_name=ablation_all_attention          --seed=$S --max_batch_length_train=500
sbatch --job-name=abl_all_attn_nov       "$SLURM" "$YAML" --boundary_mode=all         --pooling_mode=attention_no_value --experiment_name=ablation_all_attention_no_value --seed=$S --max_batch_length_train=500

# MLP
sbatch --job-name=abl_mlp_attn           "$SLURM" "$YAML" --boundary_mode=mlp         --pooling_mode=attention          --experiment_name=ablation_mlp_attention          --seed=$S
sbatch --job-name=abl_mlp_mean           "$SLURM" "$YAML" --boundary_mode=mlp         --pooling_mode=mean               --experiment_name=ablation_mlp_mean               --seed=$S
sbatch --job-name=abl_mlp_attn_nov       "$SLURM" "$YAML" --boundary_mode=mlp         --pooling_mode=attention_no_value --experiment_name=ablation_mlp_attention_no_value --seed=$S

# --- Diagnostic ablations (crash isolation) ---
sbatch --job-name=diag_no_bp_loss        "$SLURM" "$YAML" --boundary_mode=learned     --pooling_mode=attention          --experiment_name=diag_no_boundary_loss --boundary_predictor_loss_weight=0 --seed=$S
sbatch --job-name=diag_fixed_temp        "$SLURM" "$YAML" --boundary_mode=learned     --pooling_mode=attention          --experiment_name=diag_fixed_temp       --disable_temp_schedule=True       --seed=$S

echo "All ablation jobs submitted."
