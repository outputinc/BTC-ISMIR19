#!/usr/bin/env python
"""
Extract output layer representations from the BTC model.

This script extracts 25-dimensional representations from the BTC model's
output layer (logits before softmax, corresponding to chord predictions).
"""

import argparse
import os
import numpy as np
import librosa
import torch
import yaml
from scipy.ndimage import uniform_filter1d

from btc_model import BTC_model
from utils.hparams import HParams


def load_config(path):
    """Load config with proper yaml loader."""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return HParams(**config_dict)


def load_audio_segment(audio_path, sr=22050, duration=None):
    """Load audio file and optionally truncate to duration seconds."""
    if duration is not None:
        # Load only the specified duration
        wav, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)
    else:
        wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    return wav


def compute_cqt_features(wav, sr=22050, n_bins=144, bins_per_octave=24, hop_length=2048):
    """Compute CQT features from audio waveform."""
    cqt = librosa.cqt(wav, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length)
    # Log compression
    feature = np.log(np.abs(cqt) + 1e-6)
    # Transpose to [timestep, n_bins]
    feature = feature.T
    return feature


def extract_representations(audio_path, model_path='./test/btc_model.pt', duration=5.0, device=None, drop_n=False, mean_pool=None):
    """
    Extract output layer representations from BTC model.

    Args:
        audio_path: Path to audio file
        model_path: Path to model checkpoint
        duration: Duration in seconds to process (None for full audio)
        device: Torch device (auto-detected if None)
        drop_n: If True, drop the NO CHORD dimension (index 24) from output
        mean_pool: If specified, apply a mean filter with this kernel size across the time dimension of CQT

    Returns:
        representations: numpy array of shape [timestep, 25] or [timestep, 24] if drop_n=True
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config = load_config("run_config.yaml")

    # Model uses small vocab (25 chords)
    config.model['num_chords'] = 25

    # Build model
    model = BTC_model(config=config.model).to(device)

    # Load checkpoint
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load audio
    sr = config.mp3['song_hz']  # 22050
    wav = load_audio_segment(audio_path, sr=sr, duration=duration)

    # Compute CQT features
    feature = compute_cqt_features(
        wav,
        sr=sr,
        n_bins=config.feature['n_bins'],
        bins_per_octave=config.feature['bins_per_octave'],
        hop_length=config.feature['hop_length']
    )

    # Apply mean filter across time dimension if specified
    if mean_pool is not None and mean_pool > 1:
        feature = uniform_filter1d(feature, size=mean_pool, axis=0, mode='nearest')

    # Normalize
    feature = (feature - mean) / std

    # Store original number of frames before padding
    original_num_frames = feature.shape[0]

    # Pad to be divisible by timestep (108)
    n_timestep = config.model['timestep']
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    if num_pad != n_timestep:
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    else:
        num_pad = 0

    num_instances = feature.shape[0] // n_timestep

    # Extract representations
    all_representations = []
    with torch.no_grad():
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)

        for t in range(num_instances):
            start_idx = n_timestep * t
            end_idx = n_timestep * (t + 1)
            chunk = feature_tensor[:, start_idx:end_idx, :]

            # Get hidden representation from self-attention layers
            hidden_output, _ = model.self_attn_layers(chunk)

            # Pass through output layer projection to get logits (25-dim)
            logits = model.output_layer.output_projection(hidden_output)

            # logits shape: [1, timestep, 25]
            all_representations.append(logits.squeeze(0).cpu().numpy())

    # Concatenate all chunks
    representations = np.concatenate(all_representations, axis=0)

    # Remove padding frames to get original length
    representations = representations[:original_num_frames]

    # Drop NO CHORD dimension (index 24) if requested
    if drop_n:
        representations = representations[:, :24]

    return representations


def main():
    parser = argparse.ArgumentParser(description='Extract hidden representations from BTC model')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to audio file')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds (default: 5.0)')
    parser.add_argument('--output', type=str, default=None, help='Output path for .npy file (optional)')
    parser.add_argument('--model_path', type=str, default='./test/btc_model.pt', help='Path to model checkpoint')
    parser.add_argument('--drop_n', action='store_true', help='Drop the NO CHORD dimension (index 24) from output')
    parser.add_argument('--mean_pool', type=int, default=None, help='Apply mean filter with this kernel size across CQT time dimension')
    args = parser.parse_args()

    # Extract representations
    print(f"Loading audio: {args.audio_path}")
    print(f"Processing duration: {args.duration} seconds")

    representations = extract_representations(
        audio_path=args.audio_path,
        model_path=args.model_path,
        duration=args.duration,
        drop_n=args.drop_n,
        mean_pool=args.mean_pool
    )

    # Print summary
    print(f"\nRepresentation shape: {representations.shape}")
    print(f"  - Timesteps: {representations.shape[0]}")
    print(f"  - Hidden dim: {representations.shape[1]}")
    print(f"\nStatistics:")
    print(f"  - Mean: {representations.mean():.6f}")
    print(f"  - Std:  {representations.std():.6f}")
    print(f"  - Min:  {representations.min():.6f}")
    print(f"  - Max:  {representations.max():.6f}")

    # Save if output path provided
    if args.output:
        np.save(args.output, representations)
        print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
