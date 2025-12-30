#!/usr/bin/env python3
"""
Script to measure tokenization speed (tokens/second) on LibriSpeech data.

Usage:
    python test.py hparams/conformer_4x.yaml
"""

import sys
import time
import csv
from hyperpyyaml import load_hyperpyyaml


def load_transcripts_from_csv(csv_path, data_folder):
    """Load transcripts from a SpeechBrain CSV file."""
    transcripts = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'wrd' in row:
                transcripts.append(row['wrd'])
    return transcripts


def measure_tokenization_speed(tokenizer, transcripts, num_runs=3):
    """
    Measure tokenization speed over multiple runs.

    Returns:
        dict with timing statistics and tokens/second
    """
    results = []

    for run in range(num_runs):
        total_tokens = 0
        start_time = time.perf_counter()

        for transcript in transcripts:
            tokens = tokenizer.encode_as_ids(transcript)
            total_tokens += len(tokens)

        elapsed_time = time.perf_counter() - start_time
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0

        results.append({
            'run': run + 1,
            'total_tokens': total_tokens,
            'elapsed_time': elapsed_time,
            'tokens_per_second': tokens_per_second,
        })

        print(f"Run {run + 1}: {total_tokens:,} tokens in {elapsed_time:.4f}s = {tokens_per_second:,.0f} tokens/sec")

    # Calculate averages
    avg_tokens_per_second = sum(r['tokens_per_second'] for r in results) / len(results)
    avg_time = sum(r['elapsed_time'] for r in results) / len(results)
    total_tokens = results[0]['total_tokens']  # Same for all runs

    return {
        'runs': results,
        'avg_tokens_per_second': avg_tokens_per_second,
        'avg_time': avg_time,
        'total_tokens': total_tokens,
        'num_transcripts': len(transcripts),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <hparams_yaml>")
        print("Example: python test.py hparams/conformer_4x.yaml")
        sys.exit(1)

    hparams_file = sys.argv[1]

    print(f"Loading hyperparameters from {hparams_file}...")
    with open(hparams_file, encoding='utf-8') as fin:
        hparams = load_hyperpyyaml(fin)

    # Load tokenizer using pretrainer
    print("Loading tokenizer...")
    hparams['pretrainer'].collect_files()
    hparams['pretrainer'].load_collected()
    tokenizer = hparams['tokenizer']

    print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"Vocabulary size: {tokenizer.get_piece_size()}")

    # Load transcripts from various data splits
    data_folder = hparams['data_folder']
    data_manifest_folder = hparams['data_manifest_folder']

    # Test on different splits
    splits_to_test = [
        ('valid_csv', 'Validation (dev-clean)'),
    ]

    # Add test CSVs if available
    if 'test_csv' in hparams and hparams['test_csv']:
        for i, test_csv in enumerate(hparams['test_csv']):
            splits_to_test.append((test_csv, f'Test split {i+1}'))

    print("\n" + "=" * 60)
    print("TOKENIZATION SPEED TEST")
    print("=" * 60)

    all_results = {}

    for split_key, split_name in splits_to_test:
        if isinstance(split_key, str) and split_key in hparams:
            csv_path = hparams[split_key]
        else:
            csv_path = split_key

        print(f"\n--- {split_name} ---")
        print(f"CSV: {csv_path}")

        try:
            transcripts = load_transcripts_from_csv(csv_path, data_folder)
            print(f"Loaded {len(transcripts):,} transcripts")

            if len(transcripts) == 0:
                print("No transcripts found, skipping...")
                continue

            results = measure_tokenization_speed(tokenizer, transcripts)
            all_results[split_name] = results

            print(f"\nSummary for {split_name}:")
            print(f"  Transcripts: {results['num_transcripts']:,}")
            print(f"  Total tokens: {results['total_tokens']:,}")
            print(f"  Avg tokens/transcript: {results['total_tokens'] / results['num_transcripts']:.1f}")
            print(f"  Avg time: {results['avg_time']:.4f}s")
            print(f"  Avg tokens/second: {results['avg_tokens_per_second']:,.0f}")

        except FileNotFoundError as e:
            print(f"File not found: {e}")
            continue
        except Exception as e:
            print(f"Error processing {split_name}: {e}")
            continue

    # Overall summary
    if all_results:
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)

        total_transcripts = sum(r['num_transcripts'] for r in all_results.values())
        total_tokens = sum(r['total_tokens'] for r in all_results.values())
        avg_tokens_per_second = sum(r['avg_tokens_per_second'] for r in all_results.values()) / len(all_results)

        print(f"Total transcripts processed: {total_transcripts:,}")
        print(f"Total tokens generated: {total_tokens:,}")
        print(f"Average tokens/second: {avg_tokens_per_second:,.0f}")


if __name__ == "__main__":
    main()
