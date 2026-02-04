#!/usr/bin/env python
"""
Prepare MoisesDB for BTC chord recognition finetuning.

Pipeline:
1. Load MoisesDB tracks using the moisesdb library
2. Run BTC inference on mix audio to get chord labels per frame
3. Create N random submixes (1 to n_stems-1 stems) per chunk
4. Save CQT features with chord labels from mix

Download MoisesDB from: https://developer.moises.ai/research
Then set MOISESDB_PATH environment variable or use --data_path argument.
"""

import os
import json
import random
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
import torch
from tqdm import tqdm

from moisesdb.dataset import MoisesDB
from moisesdb.track import MoisesDBTrack

from btc_model import BTC_model
from utils.hparams import HParams


# Non-tonal stem types in MoisesDB
NON_TONAL_STEMS = {"drums", "percussion", "percussions"}


def load_config(config_path: str) -> HParams:
    """Load config with proper yaml loader."""
    with open(config_path, 'r') as f:
        return HParams(**yaml.load(f, Loader=yaml.FullLoader))


# Constants from BTC config
TARGET_SR = 22050
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP_LENGTH = 2048
TIMESTEP = 108  # frames per chunk
CHUNK_DURATION = 10.0  # seconds
NUM_CHORDS = 25  # major/minor vocabulary


def load_btc_model(model_path: str, config_path: str = "run_config.yaml") -> Tuple[BTC_model, np.ndarray, np.ndarray, HParams]:
    """Load pretrained BTC model and normalization statistics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    model = BTC_model(config=config.model).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, mean, std, config


def compute_cqt_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Compute CQT features matching BTC format.

    Args:
        audio: Audio waveform
        sr: Sample rate

    Returns:
        CQT features of shape [n_bins, n_frames]
    """
    cqt = librosa.cqt(
        audio,
        sr=sr,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=HOP_LENGTH
    )
    return np.abs(cqt)


def run_btc_inference(
    audio: np.ndarray,
    model: BTC_model,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """
    Run BTC inference to get per-frame chord predictions.

    Args:
        audio: Audio waveform at TARGET_SR
        model: BTC model
        mean: Normalization mean
        std: Normalization std
        device: Torch device

    Returns:
        Chord IDs array of shape [num_frames]
    """
    # Compute CQT features
    feature = compute_cqt_features(audio)
    feature = np.log(np.abs(feature) + 1e-6)  # Apply log for inference
    feature = feature.T  # [n_frames, n_bins]

    # Normalize
    feature = (feature - mean) / std

    # Pad to multiple of timestep
    n_timestep = TIMESTEP
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    if num_pad < n_timestep:
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)

    num_instance = feature.shape[0] // n_timestep

    # Run inference
    predictions = []
    with torch.no_grad():
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            chunk = feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
            self_attn_output, _ = model.self_attn_layers(chunk)
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            predictions.extend(prediction.cpu().numpy().tolist())

    # Remove padding predictions
    original_frames = feature.shape[0] - num_pad if num_pad < n_timestep else feature.shape[0]
    return np.array(predictions[:original_frames], dtype=np.int64)


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def filter_active_stems(
    stems: Dict[str, np.ndarray],
    rms_threshold: float = 0.03
) -> Dict[str, np.ndarray]:
    """Filter stems to only include those with actual audio content."""
    active_stems = {}
    for name, audio in stems.items():
        rms = compute_rms(audio)
        if rms >= rms_threshold:
            active_stems[name] = audio
    return active_stems


def is_tonal_stem(stem_name: str) -> bool:
    """Check if a stem is tonal (has harmonic content for chord recognition)."""
    stem_lower = stem_name.lower()
    for non_tonal in NON_TONAL_STEMS:
        if non_tonal in stem_lower:
            return False
    return True


def create_submixes(
    stems: Dict[str, np.ndarray],
    n_submixes: int,
    min_stems: int,
    max_stems: int,
    stem_rms_threshold: float = 0.0
) -> List[Tuple[np.ndarray, List[str]]]:
    """
    Generate random stem combinations ensuring at least one tonal stem.

    Args:
        stems: Dictionary of stem_name -> audio array
        n_submixes: Number of submixes to create
        min_stems: Minimum stems per submix
        max_stems: Maximum stems per submix
        stem_rms_threshold: Minimum RMS for individual stems

    Returns:
        List of (submix_audio, stem_names) tuples
    """
    # Filter out silent stems first
    if stem_rms_threshold > 0:
        stems = filter_active_stems(stems, stem_rms_threshold)

    if not stems:
        return []

    stem_names = list(stems.keys())
    submixes = []

    # Separate tonal and non-tonal stems
    tonal_stems = [s for s in stem_names if is_tonal_stem(s)]
    non_tonal_stems = [s for s in stem_names if not is_tonal_stem(s)]

    # If no tonal stems available, skip this chunk
    if not tonal_stems:
        return []

    for _ in range(n_submixes):
        k = random.randint(min_stems, min(max_stems, len(stem_names)))

        # Ensure at least one tonal stem
        selected = [random.choice(tonal_stems)]
        remaining_pool = [s for s in stem_names if s not in selected]

        if k > 1 and remaining_pool:
            additional = random.sample(remaining_pool, min(k - 1, len(remaining_pool)))
            selected.extend(additional)

        # Sum selected stems
        submix_audio = np.zeros_like(next(iter(stems.values())))
        for name in selected:
            submix_audio += stems[name]

        submixes.append((submix_audio, selected))

    return submixes


def get_track_stems(track: MoisesDBTrack, target_sr: int = TARGET_SR) -> Dict[str, np.ndarray]:
    """
    Extract all stems from a MoisesDB track.

    Args:
        track: MoisesDB track object
        target_sr: Target sample rate

    Returns:
        Dictionary mapping stem name to audio array
    """
    stems = {}

    # Get the stems dictionary from the track
    try:
        track_stems = track.stems
        for stem_name, stem_audio in track_stems.items():
            if stem_audio is not None and len(stem_audio) > 0:
                # Resample if needed (MoisesDB default is 44100)
                if track.sample_rate != target_sr:
                    stem_audio = librosa.resample(
                        stem_audio,
                        orig_sr=track.sample_rate,
                        target_sr=target_sr
                    )
                # Convert to mono if stereo
                if len(stem_audio.shape) > 1:
                    stem_audio = np.mean(stem_audio, axis=0)
                stems[stem_name] = stem_audio
    except Exception as e:
        print(f"Warning: Error loading stems for track {track.id}: {e}")

    return stems


def get_track_mix(track: MoisesDBTrack, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Get the mix audio from a MoisesDB track.

    Args:
        track: MoisesDB track object
        target_sr: Target sample rate

    Returns:
        Mix audio array
    """
    mix = track.audio

    # Resample if needed
    if track.sample_rate != target_sr:
        mix = librosa.resample(mix, orig_sr=track.sample_rate, target_sr=target_sr)

    # Convert to mono if stereo
    if len(mix.shape) > 1:
        mix = np.mean(mix, axis=0)

    return mix


def process_track(
    track: MoisesDBTrack,
    model: BTC_model,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    output_dir: Path,
    split: str,
    chunk_duration: float,
    submixes_per_chunk: int,
    min_stems: int,
    max_stems: int,
    stem_rms_threshold: float = 0.0
) -> Tuple[int, int]:
    """
    Process a single MoisesDB track.

    Returns:
        Tuple of (examples_created, examples_filtered)
    """
    track_id = track.id

    try:
        # Load mix and stems
        mix = get_track_mix(track, TARGET_SR)
        stems = get_track_stems(track, TARGET_SR)
    except Exception as e:
        print(f"Warning: Failed to load track {track_id}: {e}")
        return 0, 0

    if len(stems) < 2:
        return 0, 0

    # Run BTC inference on mix to get chord labels
    chord_ids = run_btc_inference(mix, model, mean, std, device)

    # Calculate chunk parameters
    samples_per_chunk = int(chunk_duration * TARGET_SR)
    frames_per_chunk = int(chunk_duration * TARGET_SR / HOP_LENGTH)

    # Ensure all stems have same length
    min_length = min(len(s) for s in stems.values())
    min_length = min(min_length, len(mix))
    stems = {k: v[:min_length] for k, v in stems.items()}

    examples_created = 0
    examples_filtered = 0
    chunk_idx = 0
    sample_offset = 0
    frame_offset = 0

    while sample_offset + samples_per_chunk <= min_length:
        # Get chord labels for this chunk
        chunk_end_frame = min(frame_offset + frames_per_chunk, len(chord_ids))
        chord_chunk = chord_ids[frame_offset:chunk_end_frame]

        if len(chord_chunk) < frames_per_chunk // 2:
            break

        # Get stem audio chunks
        stem_chunks = {
            name: audio[sample_offset:sample_offset + samples_per_chunk]
            for name, audio in stems.items()
        }

        # Create submixes
        submixes = create_submixes(
            stem_chunks, submixes_per_chunk, min_stems, max_stems, stem_rms_threshold
        )

        if not submixes:
            examples_filtered += submixes_per_chunk
            chunk_idx += 1
            sample_offset += samples_per_chunk
            frame_offset += frames_per_chunk
            continue

        for submix_idx, (submix_audio, selected_stems) in enumerate(submixes):
            # Compute CQT features for submix
            feature = compute_cqt_features(submix_audio)

            # Align feature frames to chord labels
            n_frames = feature.shape[1]
            chord_chunk_aligned = chord_chunk
            if n_frames > len(chord_chunk_aligned):
                feature = feature[:, :len(chord_chunk_aligned)]
            elif n_frames < len(chord_chunk_aligned):
                chord_chunk_aligned = chord_chunk_aligned[:n_frames]

            # Create output filename
            output_filename = f"moisesdb_{track_id}_chunk{chunk_idx}_submix{submix_idx}.pt"
            output_path = output_dir / split / output_filename

            # Save example
            example = {
                'feature': feature,
                'chord': chord_chunk_aligned.tolist(),
                'metadata': {
                    'track_id': track_id,
                    'artist': track.artist,
                    'name': track.name,
                    'genre': track.genre,
                    'stems': selected_stems,
                    'chunk_idx': chunk_idx,
                    'submix_idx': submix_idx,
                    'dataset': 'moisesdb'
                }
            }
            torch.save(example, output_path)
            examples_created += 1

        chunk_idx += 1
        sample_offset += samples_per_chunk
        frame_offset += frames_per_chunk

    return examples_created, examples_filtered


def main():
    parser = argparse.ArgumentParser(description="Prepare MoisesDB for BTC finetuning")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("MOISESDB_PATH", os.path.expanduser("~/datasets/moisesdb")),
        help="Path to MoisesDB dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/datasets/btc_finetuning_moisesdb"),
        help="Output directory for finetuning dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./test/btc_model.pt",
        help="Path to pretrained BTC model"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="run_config.yaml",
        help="Path to BTC config file"
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=10.0,
        help="Duration of each chunk in seconds"
    )
    parser.add_argument(
        "--submixes_per_chunk",
        type=int,
        default=3,
        help="Number of random submixes to create per chunk"
    )
    parser.add_argument(
        "--min_stems",
        type=int,
        default=1,
        help="Minimum number of stems in a submix"
    )
    parser.add_argument(
        "--max_stems",
        type=int,
        default=4,
        help="Maximum number of stems in a submix"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.85,
        help="Ratio of tracks to use for training (rest for validation)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks to process (for testing)"
    )
    parser.add_argument(
        "--stem_rms_threshold",
        type=float,
        default=0.03,
        help="Filter out stems with RMS below this threshold"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate for loading MoisesDB (default 44100)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Create output directories
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "valid").mkdir(parents=True, exist_ok=True)

    # Load BTC model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, mean, std, config = load_btc_model(args.model_path, args.config_path)
    print("BTC model loaded successfully")

    # Load MoisesDB
    print(f"Loading MoisesDB from: {args.data_path}")
    try:
        db = MoisesDB(data_path=args.data_path, sample_rate=args.sample_rate)
    except Exception as e:
        print(f"\nError loading MoisesDB: {e}")
        print("\nPlease download MoisesDB from: https://developer.moises.ai/research")
        print("Then either:")
        print("  1. Set MOISESDB_PATH environment variable")
        print("  2. Use --data_path argument")
        return

    n_tracks = len(db)
    print(f"Found {n_tracks} tracks in MoisesDB")

    if args.limit:
        n_tracks = min(n_tracks, args.limit)
        print(f"Limiting to {n_tracks} tracks")

    # Split into train/valid
    indices = list(range(n_tracks))
    random.shuffle(indices)
    n_train = int(n_tracks * args.train_ratio)
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]

    print(f"Train: {len(train_indices)} tracks, Valid: {len(valid_indices)} tracks")

    # Process tracks
    manifest = {
        "config": {
            "chunk_duration": args.chunk_duration,
            "submixes_per_chunk": args.submixes_per_chunk,
            "min_stems": args.min_stems,
            "max_stems": args.max_stems,
            "target_sr": TARGET_SR,
            "n_bins": N_BINS,
            "hop_length": HOP_LENGTH,
            "num_chords": NUM_CHORDS,
            "stem_rms_threshold": args.stem_rms_threshold,
            "train_ratio": args.train_ratio,
        },
        "splits": {}
    }

    for split, track_indices in [("train", train_indices), ("valid", valid_indices)]:
        print(f"\nProcessing {split} split: {len(track_indices)} tracks")

        total_examples = 0
        total_filtered = 0
        tracks_processed = 0

        for idx in tqdm(track_indices, desc=f"Processing {split}"):
            track = db[idx]
            n_examples, n_filtered = process_track(
                track, model, mean, std, device, output_dir, split,
                args.chunk_duration, args.submixes_per_chunk,
                args.min_stems, args.max_stems, args.stem_rms_threshold
            )
            total_examples += n_examples
            total_filtered += n_filtered
            if n_examples > 0:
                tracks_processed += 1

        manifest["splits"][split] = {
            "num_tracks": len(track_indices),
            "tracks_with_examples": tracks_processed,
            "num_examples": total_examples,
            "num_filtered": total_filtered
        }
        print(f"{split}: {total_examples} examples from {tracks_processed} tracks ({total_filtered} filtered)")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    print("\nMoisesDB preparation complete!")
    print(f"Output directory: {output_dir}")
    print("\nTo finetune BTC on this data, run:")
    print(f"  python finetune_btc.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
