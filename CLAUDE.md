# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of an ASR (Automatic Speech Recognition) system that extends SpeechBrain's Transformer/Conformer architecture with a novel **BoundaryPredictor** module. The system trains on LibriSpeech (960h) and uses a unique compression approach based on learned boundary detection with phoneme-based targets.

## Core Architecture

The ASR pipeline consists of four main stages:

1. **CNN Frontend** (`CNN` module) - Compresses audio features by 2-4x in time
2. **BoundaryPredictor** (`BoundaryPredictor2` class) - Novel learned compression that dynamically segments the audio representation based on phoneme boundaries
3. **Transformer/Conformer Encoder-Decoder** - Standard seq2seq ASR model
4. **CTC + Seq2Seq Joint Decoding** - Dual-loss training with beam search

### BoundaryPredictor Architecture

The `BoundaryPredictor2` module (in `BoundaryPredictor.py`) is the key innovation:

- **Input**: Hidden states from CNN (shape: `[batch, time, dim]`)
- **Boundary Detection**: Uses cosine similarity between adjacent frames to predict segment boundaries
  - Computes Q and K projections with MLP processing and residual connections
  - Samples boundaries using RelaxedBernoulli distribution during training
  - Temperature is scheduled from 1.0 → 0.0 across epochs (see `pretrain.py:357-368`)
- **Pooling**: Multi-head attention pooling with learned query vectors to aggregate each segment
- **Loss**: Binomial loss that encourages the number of predicted boundaries to match phoneme counts from the transcript (using g2p_en for grapheme-to-phoneme conversion)
- **Output**: Compressed sequence with variable-length segments

Key implementation details:

- The module maintains proper gradient flow using straight-through estimator: `hard_boundaries = (hard_samples - soft_boundaries.detach() + soft_boundaries)`
- Includes safeguards to ensure every sequence has at least one boundary
- Computes diagnostic metrics: boundary rate, coefficient of variation (CV), and adjacent boundary percentage

### Training Pipeline Flow

```
Audio → Fbank Features → Normalize → [Augmentation] → CNN →
BoundaryPredictor → Transformer Encoder-Decoder →
CTC Loss + Seq2Seq Loss + Boundary Loss
```

The training loop (`pretrain.py`) computes three losses:

1. **CTC Loss**: Connectionist Temporal Classification on encoder outputs
2. **Seq2Seq Loss**: Cross-entropy on decoder predictions
3. **Boundary Loss**: Binomial loss encouraging predicted boundaries to match phoneme counts

Loss combination: `loss = (ctc_weight * loss_ctc + (1 - ctc_weight) * loss_seq) + boundary_predictor_loss_weight * loss_boundary`

### Phoneme Target Computation

The `count_phonemes_batch()` function in `pretrain.py:80-91` converts transcripts to phoneme counts using:

- `g2p_en` library for grapheme-to-phoneme conversion
- Filters out punctuation and whitespace
- Returns float tensor of phoneme counts per utterance
- These counts are clamped to valid range `[1, actual_sequence_length]` before being used as targets

## Common Commands

### Training

```bash
# Basic training with a hyperparameter file
python pretrain.py hparams/bp_10x.yaml

# Resume training from checkpoint
python pretrain.py hparams/bp_10x.yaml --ckpt_interval_minutes=15

# Train with custom overrides
python pretrain.py hparams/bp_10x.yaml --batch_size=128 --number_of_epochs=100
```

### Testing/Evaluation

```bash
# Test tokenization speed
python test.py hparams/bp_10x.yaml
```

### Data Preparation

The LibriSpeech dataset is automatically prepared on first run via `prepare.py`. To manually trigger preparation:

```bash
python pretrain.py hparams/bp_10x.yaml --skip_prep=False
```

## Configuration Files

YAML files in `hparams/` control all model hyperparameters. Key configurations:

- `bp_10x.yaml` - BoundaryPredictor with ~10x total compression (2x CNN + ~5x BP)
- `conformer_*x.yaml` - Various compression ratios (2x, 4x, 8x, 16x, 32x)

### Important Hyperparameters

From `hparams/bp_10x.yaml`:

- `boundary_predictor_prior: 0.192` - Target compression ratio for BP (1/5.2 ≈ 0.192)
- `boundary_predictor_temp: 1.0` - Initial temperature (scheduled during training)
- `boundary_predictor_loss_weight: 1.0` - Weight for boundary prediction loss
- `use_phoneme_boundary_targets: True` - Use phoneme counting instead of prior-based targets
- `d_model: 144` - Transformer hidden dimension
- `num_encoder_layers: 12` - Conformer encoder depth
- `ctc_weight: 0` - Weight for CTC loss (0 = pure seq2seq, 1 = pure CTC)

### Data Paths

The YAML files expect:

- `data_folder`: Raw LibriSpeech audio files
- `data_manifest_folder`: Processed CSV manifests (shared across experiments)
- `output_folder`: Experiment outputs, checkpoints, logs

## Debugging Flags

The `flags.py` file contains global debug flags:

```python
PRINT_FLOW = False        # Function entry/exit logging
PRINT_DATA = False        # Tensor shapes and values
PRINT_BP_LOSS_CHECKS = True   # Boundary predictor loss diagnostics
PRINT_NAN_INF = True      # NaN/Inf detection
```

Enable these when debugging training issues, especially gradient problems or boundary prediction anomalies.

## Key Implementation Notes

### BoundaryPredictor Integration

When modifying the BoundaryPredictor:

- Changes to boundary detection logic affect `BoundaryPredictor.py:228-273`
- Pooling mechanism is in `BoundaryPredictor.py:95-211` (multi-head attention)
- Loss computation is in `BoundaryPredictor.py:536-610` and `loss.py:12-81`
- The module is called from `pretrain.py:155-216` with proper length masking

### Length Handling

The codebase carefully tracks sequence lengths through compression stages:

- Input: `wav_lens` (relative lengths 0-1)
- After CNN: lengths unchanged (CNN pads appropriately)
- After BoundaryPredictor: `bp_wav_lens` (recomputed based on boundaries)
- Lengths are properly propagated through Transformer and CTC loss

### Loss Function

The binomial loss (`loss.py`) models boundary placement as a binomial process:

- Total positions: sequence length
- Target probability: `target_boundary_counts / total_positions`
- Actual count: `num_boundaries` (sampled during training)
- Loss: negative log probability normalized by sequence length

### Temperature Scheduling

Temperature for RelaxedBernoulli sampling is scheduled in `pretrain.py:357-368`:

- Starts at 1.0 (soft, stochastic)
- Decreases linearly to near 0.0 over training
- Stays 1 step behind to prevent reaching exactly 0.0 (causes NaN)
- Set per-epoch via `BoundaryPredictor.set_temperature()`

## SpeechBrain Framework

This project uses SpeechBrain's `Brain` class pattern:

- `compute_forward()` - Forward pass through all modules
- `compute_objectives()` - Loss calculation
- `on_stage_start()` / `on_stage_end()` - Epoch hooks
- `on_fit_batch_end()` - Per-batch hooks (learning rate scheduling)

The `dataio_prepare()` function sets up data pipelines with dynamic batching for efficient GPU utilization on H100.

## Module Dependencies

- **Core**: PyTorch, SpeechBrain
- **Audio**: torchaudio (via SpeechBrain)
- **Tokenization**: sentencepiece
- **Phonemes**: g2p_en (for phoneme counting)
- **Config**: hyperpyyaml
- **Logging**: wandb (configured in YAML)

## Performance Optimization

The codebase is optimized for H100 GPUs:

- BF16 precision training (`precision: bf16`)
- Dynamic batching with `max_batch_length_train: 2000`
- Gradient accumulation support
- Multi-GPU DDP ready (via SpeechBrain)

## Checkpoint Averaging

Evaluation uses checkpoint averaging over the last N checkpoints (`avg_checkpoints: 10`). This happens automatically in `pretrain.py:335-349` via `on_evaluate_start()`.
