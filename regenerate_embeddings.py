#!/usr/bin/env python
"""
Regenerate stem_embeddings.pt using 24D output layer representations.

This script loads the existing metadata and re-extracts embeddings
using the output layer (logits) from the BTC model, dropping the
NO CHORD dimension to get 24D representations.

Keeps the full time dimension (54 timesteps) for temporal similarity analysis.
"""

import torch
import numpy as np
from pathlib import Path
import sys

from extract_representations import extract_representations

# Fixed number of timesteps for 5 seconds of audio
NUM_TIMESTEPS = 54


def pad_or_truncate(tensor, target_len=NUM_TIMESTEPS):
    """Pad or truncate tensor to target length along time dimension."""
    current_len = tensor.shape[0]
    if current_len == target_len:
        return tensor
    elif current_len < target_len:
        # Pad with zeros
        padding = torch.zeros(target_len - current_len, tensor.shape[1])
        return torch.cat([tensor, padding], dim=0)
    else:
        # Truncate
        return tensor[:target_len]


def main():
    # Load existing metadata
    print("Loading metadata...")
    metadata = torch.load('stem_metadata.pt', weights_only=False)
    print(f"Found {len(metadata)} stems to process")

    # Extract representations for each stem
    all_embeddings = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    total = len(metadata)
    for i, meta in enumerate(metadata):
        if i % 100 == 0:
            print(f"Processing {i}/{total}...", flush=True)
        stem_path = meta['stem_path']

        try:
            # Extract 24D representations (drop NO CHORD dimension)
            reps = extract_representations(
                audio_path=stem_path,
                model_path='./test/btc_model.pt',
                duration=5.0,
                device=device,
                drop_n=True,  # Drop NO CHORD to get 24D
                mean_pool=None
            )

            # reps shape: [timesteps, 24]
            reps_tensor = torch.tensor(reps, dtype=torch.float32)

            # Pad or truncate to fixed length
            reps_tensor = pad_or_truncate(reps_tensor, NUM_TIMESTEPS)

            all_embeddings.append(reps_tensor)

        except Exception as e:
            print(f"\nError processing {stem_path}: {e}")
            # Use zeros as fallback
            all_embeddings.append(torch.zeros(NUM_TIMESTEPS, 24))

    # Stack all embeddings into single tensor [N, 54, 24]
    embeddings = torch.stack(all_embeddings)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std:  {embeddings.std():.4f}")

    # Save
    output = {
        'embeddings': embeddings,  # [N, 54, 24]
    }

    torch.save(output, 'stem_embeddings.pt')
    print("\nSaved to stem_embeddings.pt")


if __name__ == '__main__':
    main()
