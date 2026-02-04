#!/usr/bin/env python
"""
Generate embeddings for slakh stems using both base and finetuned BTC models.

This creates two embedding sets for comparison:
- Base model: ./test/btc_model.pt
- Finetuned model: ./finetuned_models/btc_finetuned_best.pt
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import random

from extract_representations import extract_representations

# Fixed number of timesteps for 5 seconds of audio
NUM_TIMESTEPS = 54
NUM_STEMS = 1000  # Target number of stems to embed

# Slakh dataset path
SLAKH_BASE = Path("/home/ubuntu/datasets/slakh2100_contrastive/original/slakh2100_redux_16k/train")


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


def collect_slakh_metadata(max_stems=NUM_STEMS):
    """Collect metadata for slakh stems."""
    metadata = []

    # Get all track directories
    track_dirs = sorted([d for d in SLAKH_BASE.iterdir() if d.is_dir() and d.name.startswith("Track")])
    print(f"Found {len(track_dirs)} tracks")

    for track_dir in track_dirs:
        metadata_file = track_dir / "metadata.yaml"
        stems_dir = track_dir / "stems"

        if not metadata_file.exists() or not stems_dir.exists():
            continue

        # Parse metadata
        try:
            with open(metadata_file, 'r') as f:
                content = f.read().strip()
                track_meta = yaml.safe_load(content)
        except Exception as e:
            print(f"Warning: Failed to parse {metadata_file}: {e}")
            continue

        if track_meta is None:
            continue

        stems_info = track_meta.get('stems', {})

        for stem_id, stem_info in stems_info.items():
            # Skip stems that weren't rendered
            if not stem_info.get('audio_rendered', False):
                continue

            stem_path = stems_dir / f"{stem_id}.flac"
            if not stem_path.exists():
                continue

            # Get instrument info
            inst_class = stem_info.get('inst_class', 'Unknown')
            midi_program = stem_info.get('midi_program_name', 'Unknown')

            metadata.append({
                'track_name': track_dir.name,
                'stem_name': f"{stem_id}_{inst_class}",
                'stem_path': str(stem_path),
                'inst_class': inst_class,
                'midi_program': midi_program,
                'is_drum': stem_info.get('is_drum', False),
            })

            if len(metadata) >= max_stems:
                return metadata

    return metadata


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collect slakh metadata
    print("Collecting slakh metadata...")
    metadata = collect_slakh_metadata(NUM_STEMS)
    print(f"Collected {len(metadata)} stems from {len(set(m['track_name'] for m in metadata))} tracks")

    # Print some stats
    inst_classes = {}
    for m in metadata:
        inst = m['inst_class']
        inst_classes[inst] = inst_classes.get(inst, 0) + 1
    print("\nInstrument distribution:")
    for inst, count in sorted(inst_classes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {inst}: {count}")

    # Model paths
    base_model_path = './test/btc_model.pt'
    finetuned_model_path = './finetuned_models/btc_finetuned_best.pt'

    # Extract base model embeddings
    print("\n" + "="*50)
    print("Extracting BASE model embeddings...")
    print("="*50)
    base_embeddings = extract_embeddings_for_model(
        metadata, base_model_path, device, "BASE"
    )
    print(f"Base embeddings shape: {base_embeddings.shape}")

    # Extract finetuned model embeddings
    print("\n" + "="*50)
    print("Extracting FINETUNED model embeddings...")
    print("="*50)
    finetuned_embeddings = extract_embeddings_for_model(
        metadata, finetuned_model_path, device, "FINETUNED"
    )
    print(f"Finetuned embeddings shape: {finetuned_embeddings.shape}")

    # Save comparison embeddings
    output = {
        'base_embeddings': base_embeddings,        # [N, 54, 24]
        'finetuned_embeddings': finetuned_embeddings,  # [N, 54, 24]
        'metadata': metadata,
    }

    output_path = 'comparison_embeddings_slakh.pt'
    torch.save(output, output_path)
    print(f"\nSaved to {output_path}")

    # Print statistics
    print("\nBase model embeddings statistics:")
    print(f"  Mean: {base_embeddings.mean():.4f}")
    print(f"  Std:  {base_embeddings.std():.4f}")

    print("\nFinetuned model embeddings statistics:")
    print(f"  Mean: {finetuned_embeddings.mean():.4f}")
    print(f"  Std:  {finetuned_embeddings.std():.4f}")


if __name__ == '__main__':
    main()
