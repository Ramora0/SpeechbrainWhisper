#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/transformer.yaml
> python train.py hparams/conformer.yaml

With the default hyperparameters, the system employs a convolutional frontend and a transformer.
The decoder is based on a Transformer decoder. Beamsearch coupled with a Transformer
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The best model is the average of the checkpoints from last 5 epochs.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020, 2021, 2022
 * Titouan Parcollet 2021, 2022
"""

import os
import sys
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

import flags
import string
from g2p_en import G2p

logger = get_logger(__name__)

# Initialize G2P converter once at module level
G2P_CONVERTER = G2p()


def count_phonemes(text: str) -> float:
    """Count phonemes in text using G2P conversion.

    Args:
        text: Input text string

    Returns:
        Number of phonemes (as float for consistency with loss function)
    """
    normalized = text.strip().lower()
    if not normalized:
        return 0.0

    phoneme_sequence = G2P_CONVERTER(normalized)
    phoneme_tokens = [
        token
        for token in phoneme_sequence
        if token.strip() and token not in string.punctuation
    ]
    return float(len(phoneme_tokens))


def count_phonemes_batch(text_list: list) -> torch.Tensor:
    """Count phonemes for a batch of texts.

    Args:
        text_list: List of text strings

    Returns:
        Tensor of shape (batch_size,) with phoneme counts
    """
    counts = [count_phonemes(text) for text in text_list]
    return torch.tensor(counts, dtype=torch.float32)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        if flags.PRINT_NAN_INF:
            print(
                f"[DEBUG] After compute_features: feats has NaN: {torch.isnan(feats).any()}, shape: {feats.shape}")

        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)
        if flags.PRINT_NAN_INF:
            print(
                f"[DEBUG] After normalize: feats has NaN: {torch.isnan(feats).any()}")

        # Add feature augmentation if specified.
        augment_warmup = 0
        if hasattr(self.hparams, "augment_warmup"):
            augment_warmup = self.hparams.augment_warmup
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            if self.optimizer_step > augment_warmup:
                feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
                tokens_bos = self.hparams.fea_augment.replicate_labels(
                    tokens_bos
                )
                if flags.PRINT_NAN_INF:
                    print(
                        f"[DEBUG] After fea_augment: feats has NaN: {torch.isnan(feats).any()}")

        # forward modules
        enc = self.modules.CNN(feats)
        enc_lens = wav_lens

        if flags.PRINT_NAN_INF:
            print(
                f"[DEBUG] After CNN: enc has NaN: {torch.isnan(enc).any()}, shape: {enc.shape}")

        # Reshape CNN output from 4D to 3D (same logic as TransformerASR.encode)
        # CNN outputs (batch, time, ch1, ch2) and we reshape to (batch, time, ch1*ch2)
        if enc.dim() == 4:
            bz, t, ch1, ch2 = enc.shape
            enc = enc.reshape(bz, t, ch1 * ch2)

        if flags.PRINT_NAN_INF:
            print(
                f"[DEBUG] After reshape: enc has NaN: {torch.isnan(enc).any()}, has Inf: {torch.isinf(enc).any()}")
        if torch.isnan(enc).any() or torch.isinf(enc).any():
            if flags.PRINT_NAN_INF:
                print(f"[DEBUG] enc stats: min={enc[~torch.isnan(enc) & ~torch.isinf(enc)].min() if (~torch.isnan(enc) & ~torch.isinf(enc)).any() else 'N/A'}, "
                      f"max={enc[~torch.isnan(enc) & ~torch.isinf(enc)].max() if (~torch.isnan(enc) & ~torch.isinf(enc)).any() else 'N/A'}")
                print(
                    f"[DEBUG] Number of NaN values: {torch.isnan(enc).sum()}, Inf values: {torch.isinf(enc).sum()}")
            # Check CNN parameters
            for name, param in self.modules.CNN.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    if flags.PRINT_NAN_INF:
                        print(
                            f"[DEBUG] CNN parameter {name} has NaN: {torch.isnan(param).any()}, Inf: {torch.isinf(param).any()}")

        # === BoundaryPredictor Integration ===
        # Initialize BP variables (so code works even if BP block is commented out)
        bp_loss = torch.tensor(0.0, device=enc.device)
        num_boundaries = 0
        total_positions = 0
        boundary_cv = None
        boundary_adjacent_pct = None

        # Apply BoundaryPredictor
        if flags.PRINT_FLOW:
            print(f"[pretrain.py] BEFORE BoundaryPredictor:")
        if flags.PRINT_DATA:
            print(f"  enc.shape = {enc.shape}")
            print(f"  enc_lens.shape = {enc_lens.shape}")
            print(f"  enc_lens = {enc_lens}")

        # Compute target boundary counts

        # Option 1: Phoneme-based targets (old approach - commented out)
        # if stage == sb.Stage.TRAIN:
        #     # Count phonemes from text
        #     target_boundary_counts = count_phonemes_batch(batch.wrd)
        #
        #     if flags.PRINT_DATA:
        #         print(f"[Phoneme Targets] text: {batch.wrd[0][:50]}...")
        #         print(
        #             f"[Phoneme Targets] target_counts: {target_boundary_counts[:5]}")
        # else:
        #     target_boundary_counts = None

        # Option 2: Prior-based targets (current approach)
        # The prior is the fraction of positions that should be boundaries:
        # - prior = 1.0 means no compression (1 boundary per position)
        # - prior = 0.5 means 2x compression (1 boundary per 2 positions)
        # - prior = 0.125 means 8x compression (1 boundary per 8 positions)
        # target_counts = total_positions * prior
        if stage == sb.Stage.TRAIN:
            prior = self.hparams.boundary_predictor_prior
            batch_size, seq_len, _ = enc.shape
            # Total positions for each sequence (accounting for variable lengths)
            total_positions = (enc_lens * seq_len).float()
            # Target boundary counts based on prior
            target_boundary_counts = (
                total_positions * prior).round().clamp(min=1.0)

            if flags.PRINT_DATA:
                print(f"[Target Boundaries] prior: {prior}")
                print(
                    f"[Target Boundaries] total_positions: {total_positions[:5]}")
                print(
                    f"[Target Boundaries] target_counts: {target_boundary_counts[:5]}")
        else:
            target_boundary_counts = None

        # Apply BoundaryPredictor with lengths (overwrites enc and enc_lens)
        (enc, bp_loss, num_boundaries, total_positions,
         enc_lens, boundary_cv, boundary_adjacent_pct) = self.modules.BoundaryPredictor( # TODO: Ablate fixed pooling
            hidden=enc,
            lengths=enc_lens,
            target_boundary_counts=target_boundary_counts,
            return_unreduced_boundary_loss=False
        )

        if flags.PRINT_FLOW:
            print(f"[pretrain.py] AFTER BoundaryPredictor:")
        if flags.PRINT_DATA:
            print(f"  enc.shape = {enc.shape}")
            print(f"  enc_lens.shape = {enc_lens.shape}")
            print(f"  enc_lens = {enc_lens}")
            print(f"  num_boundaries = {num_boundaries}")
            print(f"  total_positions = {total_positions}")

        # Store BP outputs for logging
        self.boundary_predictor_loss = bp_loss
        self.num_boundaries = num_boundaries
        self.total_positions = total_positions
        self.boundary_cv = boundary_cv
        self.boundary_adjacent_pct = boundary_adjacent_pct
        # === End BoundaryPredictor Integration ===

        # Explicitly zero out padding positions
        # batch_size, seq_len, hidden_dim = enc.shape
        # # Create mask: True for valid positions, False for padding
        # lengths_abs = (enc_lens * seq_len).long()
        # mask = torch.arange(seq_len, device=enc.device).unsqueeze(
        #     0) < lengths_abs.unsqueeze(1)
        # # Expand mask to match hidden dimension and zero out padding
        # mask = mask.unsqueeze(-1).expand_as(enc)
        # enc = enc * mask.float()

        if flags.PRINT_FLOW:
            print(f"[pretrain.py] BEFORE Transformer:")
        if flags.PRINT_DATA:
            print(f"  enc.shape = {enc.shape}")
            print(f"  tokens_bos.shape = {tokens_bos.shape}")
            print(f"  enc_lens.shape = {enc_lens.shape}")
            print(f"  enc_lens = {enc_lens}")
            print(
                f"  enc_lens min/max = {enc_lens.min().item():.4f} / {enc_lens.max().item():.4f}")

        enc_out, pred = self.modules.Transformer(
            enc, tokens_bos, enc_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if any([is_valid_search, is_test_search]):
            # Note: For valid_search, for the sake of efficiency, we only perform beamsearch with
            # limited capacity and no LM to give user some idea of how the AM is doing

            # Decide searcher for inference: valid or test search
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(
                    enc_out.detach(), enc_lens
                )
            else:
                hyps, _, _, _ = self.hparams.test_search(
                    enc_out.detach(), enc_lens
                )

        return p_ctc, p_seq, enc_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, enc_lens, hyps) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN:
            # Labels must be extended if parallel augmentation or concatenated
            # augmentation was performed on the input (increasing the time dimension)
            augment_warmup = 0
            if hasattr(self.hparams, "augment_warmup"):
                augment_warmup = self.hparams.augment_warmup
            if (
                hasattr(self.hparams, "fea_augment")
                and self.optimizer_step > augment_warmup
            ):
                (
                    tokens,
                    tokens_lens,
                    tokens_eos,
                    tokens_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, tokens_lens, tokens_eos, tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, enc_lens, tokens_lens
        ).sum()

        # Add boundary predictor loss
        if stage == sb.Stage.TRAIN:
            loss_boundary = self.boundary_predictor_loss
            target_compression = 1.0 / self.hparams.boundary_predictor_prior
            # Calculate actual compression: original_length / compressed_length
            actual_compression = self.total_positions / self.num_boundaries
            print(f"Binomial loss: {loss_boundary.item():.6f}, Target compression: {target_compression:.2f}x, Actual compression: {actual_compression:.2f}x")
        else:
            loss_boundary = torch.tensor(0.0, device=p_ctc.device)

        # Combine losses
        loss_asr = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        loss = (
            loss_asr
            + self.hparams.boundary_predictor_loss_weight * loss_boundary
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint average if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        if flags.PRINT_FLOW:
            print("Loaded the average")

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
        else:
            # Schedule temperature from 1.0 to 0.0 over training
            # Keep temperature 1 step behind to prevent it from reaching 0 (which causes NaN)
            total_epochs = self.hparams.number_of_epochs
            if total_epochs > 1:
                # Clamp epoch to stay 1 step behind, preventing temperature from reaching 0
                effective_epoch = min(epoch, total_epochs - 2)
                temperature = 1.0 - (effective_epoch / (total_epochs - 1))
            else:
                temperature = 1.0
            self.modules.BoundaryPredictor.set_temperature(temperature)
            logger.info(
                f"Epoch {epoch}: BoundaryPredictor temperature = {temperature:.4f}")

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            # Add boundary predictor statistics
            if hasattr(self, 'num_boundaries') and hasattr(self, 'total_positions'):
                if self.total_positions > 0:
                    boundary_rate = self.num_boundaries / self.total_positions
                    stage_stats["boundary_rate"] = boundary_rate
                if self.boundary_cv is not None:
                    stage_stats["boundary_cv"] = self.boundary_cv
                if self.boundary_adjacent_pct is not None:
                    stage_stats["boundary_adjacent_pct"] = self.boundary_adjacent_pct
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

            # Add boundary predictor statistics
            if hasattr(self, 'num_boundaries') and hasattr(self, 'total_positions'):
                if self.total_positions > 0:
                    boundary_rate = self.num_boundaries / self.total_positions
                    stage_stats["boundary_rate"] = boundary_rate
                if self.boundary_cv is not None:
                    stage_stats["boundary_cv"] = self.boundary_cv
                if self.boundary_adjacent_pct is not None:
                    stage_stats["boundary_adjacent_pct"] = self.boundary_adjacent_pct

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            # Check for NaN/Inf in gradients before optimizer step
            for name, param in self.modules.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        if flags.PRINT_NAN_INF:
                            print(
                                f"[DEBUG] Gradient issue in {name}: NaN: {torch.isnan(param.grad).any()}, Inf: {torch.isinf(param.grad).any()}")
                            print(f"[DEBUG] Grad norm: {param.grad.norm()}")

            # Check for NaN/Inf in parameters after optimizer step
            self.hparams.noam_annealing(self.optimizer)

            for name, param in self.modules.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    if flags.PRINT_NAN_INF:
                        print(
                            f"[DEBUG] Parameter corrupted after optimizer step: {name}")


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)

            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing Librispeech)
    from prepare import prepare_librispeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["data_manifest_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            max_key="ACC",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
