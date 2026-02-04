#!/usr/bin/env python
"""
Generate embeddings for stems using both base and finetuned BTC models.

This creates embeddings for comparison:
- Base model: ./test/btc_model.pt
- Finetuned model: ./finetuned_models/btc_finetuned_best.pt (COCO+Slakh, 71.92% val acc)

Supports both COCO Chorales and Slakh2100 datasets.
"""

import torch
import numpy as np
from pathlib import Path
import random
import os
import argparse
import yaml

from extract_representations import extract_representations

# Fixed number of timesteps for 5 seconds of audio
NUM_TIMESTEPS = 54

# Dataset paths
COCO_DIR = os.path.expanduser("~/datasets/coco_chorales_contrastive/original")
SLAKH_DIR = os.path.expanduser("~/datasets/slakh2100_contrastive/original/slakh2100_redux_16k")


def pad_or_truncate(tensor, target_len=NUM_TIMESTEPS):
    """Pad or truncate tensor to target length along time dimension."""
    current_len = tensor.shape[0]
    if current_len == target_len:
        return tensor
    elif current_len < target_len:
        padding = torch.zeros(target_len - current_len, tensor.shape[1])
        return torch.cat([tensor, padding], dim=0)
    else:
        return tensor[:target_len]


def get_coco_stems(coco_dir, split="valid", max_stems=500):
    """Get stem paths from COCO Chorales."""
    stems = []
    split_dir = Path(coco_dir) / split

    if not split_dir.exists():
        print(f"COCO split dir not found: {split_dir}")
        return stems

    for track_dir in sorted(split_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        stems_dir = track_dir / "stems_audio"
        if not stems_dir.exists():
            continue

        for stem_path in sorted(stems_dir.glob("*.wav")):
            if stem_path.name.startswith("._"):
                continue
            stems.append({
                'stem_path': str(stem_path),
                'stem_name': stem_path.stem,
                'track_name': track_dir.name,
                'dataset': 'coco',
            })

    random.shuffle(stems)
    return stems[:max_stems]


def check_audio_not_silent(path, duration=5.0, sr=22050, rms_threshold=0.03):
    """Check if audio has actual content (not silent).

    Uses higher threshold (0.03) to filter out mostly-silent stems.
    """
    try:
        import librosa
        wav, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
        rms = np.sqrt(np.mean(wav**2))
        return rms > rms_threshold
    except:
        return False


def load_slakh_metadata(track_dir):
    """Load metadata.yaml for a Slakh track."""
    metadata_path = Path(track_dir) / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
    return None


# Non-tonal instrument classes to exclude
NON_TONAL_CLASSES = {'Drums', 'Sound Effects', 'Percussion'}


def get_slakh_stems(slakh_dir, split="validation", max_stems=500, filter_silent=True, filter_nontonal=True):
    """Get stem paths from Slakh2100.

    Args:
        slakh_dir: Path to Slakh dataset
        split: Dataset split (train/validation/test)
        max_stems: Maximum number of stems to return
        filter_silent: Filter out stems with low RMS (mostly silent)
        filter_nontonal: Filter out drums and other non-tonal instruments
    """
    stems = []
    split_dir = Path(slakh_dir) / split

    if not split_dir.exists():
        print(f"Slakh split dir not found: {split_dir}")
        return stems

    all_candidates = []
    skipped_drums = 0
    skipped_nontonal = 0

    for track_dir in sorted(split_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        stems_dir = track_dir / "stems"
        if not stems_dir.exists():
            continue

        # Load metadata for instrument filtering
        metadata = load_slakh_metadata(track_dir) if filter_nontonal else None
        stem_info = metadata.get('stems', {}) if metadata else {}

        for stem_path in sorted(stems_dir.glob("*.flac")):
            if stem_path.name.startswith("._"):
                continue

            stem_id = stem_path.stem  # e.g., "S00", "S01"

            # Filter out drums and non-tonal instruments using metadata
            if filter_nontonal and stem_id in stem_info:
                info = stem_info[stem_id]
                if info.get('is_drum', False):
                    skipped_drums += 1
                    continue
                inst_class = info.get('inst_class', '')
                if inst_class in NON_TONAL_CLASSES:
                    skipped_nontonal += 1
                    continue

            all_candidates.append({
                'stem_path': str(stem_path),
                'stem_name': stem_id,
                'track_name': track_dir.name,
                'dataset': 'slakh',
                'inst_class': stem_info.get(stem_id, {}).get('inst_class', 'Unknown') if stem_info else 'Unknown',
            })

    if filter_nontonal:
        print(f"Filtered out {skipped_drums} drum stems and {skipped_nontonal} non-tonal stems")

    random.shuffle(all_candidates)

    # Filter out silent stems
    if filter_silent:
        print(f"Filtering silent Slakh stems (checking up to {min(len(all_candidates), max_stems * 3)} candidates)...")
        for candidate in all_candidates[:max_stems * 3]:  # Check 3x to account for filtering
            if check_audio_not_silent(candidate['stem_path']):
                stems.append(candidate)
                if len(stems) >= max_stems:
                    break
            if len(stems) % 50 == 0 and len(stems) > 0:
                print(f"  Found {len(stems)} non-silent stems...")
    else:
        stems = all_candidates[:max_stems]

    return stems


def extract_embeddings_for_model(metadata, model_path, device, model_name="model"):
    """Extract embeddings for all stems using a specific model."""
    all_embeddings = []
    total = len(metadata)

    for i, meta in enumerate(metadata):
        if i % 100 == 0:
            print(f"[{model_name}] Processing {i}/{total}...", flush=True)

        try:
            reps = extract_representations(
                audio_path=meta['stem_path'],
                model_path=model_path,
                duration=5.0,
                device=device,
                drop_n=True,  # Drop NO CHORD to get 24D
                mean_pool=None
            )

            reps_tensor = torch.tensor(reps, dtype=torch.float32)
            reps_tensor = pad_or_truncate(reps_tensor, NUM_TIMESTEPS)
            all_embeddings.append(reps_tensor)

        except Exception as e:
            print(f"\nError processing {meta['stem_path']}: {e}")
            all_embeddings.append(torch.zeros(NUM_TIMESTEPS, 24))

    return torch.stack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Generate comparison embeddings for BTC demo")
    parser.add_argument("--coco_stems", type=int, default=500, help="Number of COCO stems to include")
    parser.add_argument("--slakh_stems", type=int, default=500, help="Number of Slakh stems to include")
    parser.add_argument("--output", type=str, default="comparison_embeddings_combined.pt", help="Output file")
    parser.add_argument("--base_model", type=str, default="./test/btc_model.pt", help="Base model path")
    parser.add_argument("--finetuned_model", type=str, default="./finetuned_models/btc_finetuned_best.pt", help="Finetuned model path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collect stems from both datasets
    print("\nCollecting stems...")
    metadata = []

    coco_stems = get_coco_stems(COCO_DIR, split="valid", max_stems=args.coco_stems)
    print(f"Found {len(coco_stems)} COCO stems")
    metadata.extend(coco_stems)

    slakh_stems = get_slakh_stems(SLAKH_DIR, split="validation", max_stems=args.slakh_stems)
    print(f"Found {len(slakh_stems)} Slakh stems")
    metadata.extend(slakh_stems)

    print(f"\nTotal stems: {len(metadata)}")
    print(f"  COCO: {len([m for m in metadata if m['dataset'] == 'coco'])}")
    print(f"  Slakh: {len([m for m in metadata if m['dataset'] == 'slakh'])}")

    # Extract base model embeddings
    print("\n" + "="*50)
    print("Extracting BASE model embeddings...")
    print(f"Model: {args.base_model}")
    print("="*50)
    base_embeddings = extract_embeddings_for_model(
        metadata, args.base_model, device, "BASE"
    )
    print(f"Base embeddings shape: {base_embeddings.shape}")

    # Extract finetuned model embeddings
    print("\n" + "="*50)
    print("Extracting FINETUNED model embeddings...")
    print(f"Model: {args.finetuned_model}")
    print("="*50)
    finetuned_embeddings = extract_embeddings_for_model(
        metadata, args.finetuned_model, device, "FINETUNED"
    )
    print(f"Finetuned embeddings shape: {finetuned_embeddings.shape}")

    # Save comparison embeddings
    output = {
        'base_embeddings': base_embeddings,
        'finetuned_embeddings': finetuned_embeddings,
        'metadata': metadata,
        'base_model': args.base_model,
        'finetuned_model': args.finetuned_model,
    }

    torch.save(output, args.output)
    print(f"\nSaved to {args.output}")

    # Print statistics
    print("\nBase model embeddings statistics:")
    print(f"  Mean: {base_embeddings.mean():.4f}")
    print(f"  Std:  {base_embeddings.std():.4f}")

    print("\nFinetuned model embeddings statistics:")
    print(f"  Mean: {finetuned_embeddings.mean():.4f}")
    print(f"  Std:  {finetuned_embeddings.std():.4f}")


if __name__ == '__main__':
    main()
