#!/usr/bin/env python
"""
Create finetuning dataset for BTC chord recognition using COCO Chorales, Slakh2100, and/or MUSDB18.

Pipeline:
1. Run BTC inference on mix audio to get chord labels per frame
2. Create N random submixes (1 to n_stems-1 stems) per chunk
3. Save CQT features with chord labels from mix

Supported datasets:
- COCO Chorales: {split}/track_id/mix.wav, stems in stems_audio/*.wav
- Slakh2100: slakh2100_redux_16k/{split}/TrackXXXXX/mix.flac, stems in stems/*.flac
- MUSDB18: {split}/track_name/mixture.wav, stems: bass.wav, drums.wav, other.wav, vocals.wav
"""

import os
import json
import random
import argparse
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
import torch
from tqdm import tqdm

from btc_model import BTC_model
from utils.hparams import HParams


# Non-tonal instrument classes in Slakh2100 that shouldn't be used alone
NON_TONAL_CLASSES = {"drums", "sound effects", "percussive"}

# MUSDB18 stem names and their tonality
MUSDB_STEMS = ["bass", "drums", "other", "vocals"]
MUSDB_STEM_TONALITY = {
    "bass": True,
    "drums": False,  # Non-tonal
    "other": True,
    "vocals": True,
}


def load_config(config_path: str) -> HParams:
    """Load config with proper yaml loader."""
    with open(config_path, 'r') as f:
        return HParams(**yaml.load(f, Loader=yaml.FullLoader))


# Constants from BTC config
TARGET_SR = 22050
SOURCE_SR = 16000
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
    # Return magnitude (log is applied on load in AudioDataset)
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


def get_track_paths_coco(coco_dir: Path, split: str) -> List[Path]:
    """Get track paths for COCO Chorales."""
    split_dir = coco_dir / split
    if not split_dir.exists():
        return []
    # Filter to only include directories (exclude .DS_Store, etc.)
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def get_track_paths_slakh(slakh_dir: Path, split: str) -> List[Path]:
    """Get track paths for Slakh2100."""
    # Handle the nested slakh2100_redux_16k folder
    slakh_root = slakh_dir / "slakh2100_redux_16k"
    split_dir = slakh_root / split
    if not split_dir.exists():
        return []
    # Filter to only include directories (exclude .DS_Store, etc.)
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def get_track_paths_musdb(musdb_dir: Path, split: str) -> List[Path]:
    """Get track paths for MUSDB18."""
    split_dir = musdb_dir / split
    if not split_dir.exists():
        return []
    # Filter to only include directories (exclude .DS_Store, etc.)
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def get_mix_path(track_path: Path, dataset_type: str) -> Path:
    """Get mix audio path based on dataset type."""
    if dataset_type == "coco":
        return track_path / "mix.wav"
    elif dataset_type == "musdb":
        return track_path / "mixture.wav"
    else:  # slakh
        return track_path / "mix.flac"


def get_stems_dir(track_path: Path, dataset_type: str) -> Path:
    """Get stems directory based on dataset type."""
    if dataset_type == "coco":
        return track_path / "stems_audio"
    elif dataset_type == "musdb":
        return track_path  # MUSDB stems are in the same directory as mixture.wav
    else:  # slakh
        return track_path / "stems"


def load_slakh_metadata(track_path: Path) -> Dict[str, bool]:
    """
    Load Slakh metadata and determine which stems are tonal.

    Args:
        track_path: Path to track directory

    Returns:
        Dictionary mapping stem name (e.g., "S00") to is_tonal boolean
    """
    metadata_path = track_path / "metadata.yaml"
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        print(f"Warning: Failed to parse metadata for {track_path.name}: {e}")
        return {}

    stem_tonality = {}
    stems_info = metadata.get('stems', {})

    for stem_name, stem_data in stems_info.items():
        # Check if drum
        is_drum = stem_data.get('is_drum', False)
        # Check instrument class (case-insensitive)
        inst_class = stem_data.get('inst_class', '').lower()
        is_non_tonal = is_drum or inst_class in NON_TONAL_CLASSES

        stem_tonality[stem_name] = not is_non_tonal

    return stem_tonality


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) of audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_rms_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """Compute RMS in decibels."""
    rms = compute_rms(audio)
    if rms > 0:
        return 20 * np.log10(rms / ref)
    return -np.inf


def filter_active_stems(
    stems: Dict[str, np.ndarray],
    rms_threshold: float = 0.03
) -> Dict[str, np.ndarray]:
    """
    Filter stems to only include those with actual audio content.

    Args:
        stems: Dictionary of stem_name -> audio array (chunk)
        rms_threshold: Minimum RMS for a stem to be considered active

    Returns:
        Dictionary containing only active stems
    """
    active_stems = {}
    for name, audio in stems.items():
        rms = compute_rms(audio)
        if rms >= rms_threshold:
            active_stems[name] = audio
    return active_stems


def create_submixes(
    stems: Dict[str, np.ndarray],
    n_submixes: int,
    min_stems: int,
    max_stems: int,
    stem_tonality: Optional[Dict[str, bool]] = None,
    stem_rms_threshold: float = 0.0
) -> List[Tuple[np.ndarray, List[str]]]:
    """
    Generate random stem combinations.

    For Slakh2100, ensures at least one tonal stem is included in each submix
    to maintain harmonic context for chord recognition.

    Stems are filtered by RMS threshold to exclude silent/quiet stems.

    Args:
        stems: Dictionary of stem_name -> audio array
        n_submixes: Number of submixes to create
        min_stems: Minimum stems per submix
        max_stems: Maximum stems per submix
        stem_tonality: Optional dict mapping stem_name -> is_tonal (for Slakh2100)
        stem_rms_threshold: Minimum RMS for individual stems (0.0 = no filtering)

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

    # Separate tonal and non-tonal stems if tonality info provided
    if stem_tonality:
        tonal_stems = [s for s in stem_names if stem_tonality.get(s, True)]
        non_tonal_stems = [s for s in stem_names if not stem_tonality.get(s, True)]
    else:
        tonal_stems = stem_names
        non_tonal_stems = []

    # If no tonal stems available, skip this chunk
    if not tonal_stems:
        return []

    for _ in range(n_submixes):
        k = random.randint(min_stems, min(max_stems, len(stem_names)))

        if stem_tonality and non_tonal_stems:
            # Ensure at least one tonal stem
            # Pick 1 tonal stem first, then fill remaining from all stems
            selected = [random.choice(tonal_stems)]
            remaining_pool = [s for s in stem_names if s not in selected]

            if k > 1 and remaining_pool:
                additional = random.sample(remaining_pool, min(k - 1, len(remaining_pool)))
                selected.extend(additional)
        else:
            # No tonality constraints (COCO or all tonal)
            selected = random.sample(stem_names, min(k, len(stem_names)))

        # Sum selected stems
        submix_audio = np.zeros_like(next(iter(stems.values())))
        for name in selected:
            submix_audio += stems[name]

        submixes.append((submix_audio, selected))

    return submixes


def process_track(
    track_path: Path,
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
    dataset_type: str = "coco",
    stem_rms_threshold: float = 0.0
) -> Tuple[int, int]:
    """
    Process a single track end-to-end.

    Args:
        track_path: Path to track directory
        model: BTC model
        mean: Normalization mean
        std: Normalization std
        device: Torch device
        output_dir: Output directory
        split: Dataset split (train/valid)
        chunk_duration: Duration of each chunk in seconds
        submixes_per_chunk: Number of submixes per chunk
        min_stems: Minimum stems per submix
        max_stems: Maximum stems per submix
        dataset_type: "coco", "slakh", or "musdb"
        stem_rms_threshold: Filter individual stems with RMS below this (0.0 = no filtering)

    Returns:
        Tuple of (examples_created, examples_filtered)
    """
    track_id = track_path.name
    mix_path = get_mix_path(track_path, dataset_type)
    stems_dir = get_stems_dir(track_path, dataset_type)

    if not mix_path.exists() or not stems_dir.exists():
        return 0, 0

    # Load and resample mix (16kHz -> 22kHz)
    mix, _ = librosa.load(mix_path, sr=TARGET_SR, mono=True)

    # Run BTC inference on mix to get chord labels
    chord_ids = run_btc_inference(mix, model, mean, std, device)

    # Load and resample all stems (supports both .wav and .flac)
    stems = {}
    if dataset_type == "musdb":
        # MUSDB has fixed stem names in the same directory
        for stem_name in MUSDB_STEMS:
            stem_path = stems_dir / f"{stem_name}.wav"
            if stem_path.exists():
                stem_audio, _ = librosa.load(stem_path, sr=TARGET_SR, mono=True)
                stems[stem_name] = stem_audio
    else:
        # COCO and Slakh: iterate over stems directory
        for stem_path in stems_dir.iterdir():
            # Filter out non-audio files and macOS resource forks
            if stem_path.name.startswith("._"):
                continue
            if stem_path.suffix.lower() not in [".wav", ".flac"]:
                continue
            stem_name = stem_path.stem  # e.g., "1_trumpet" or "S00"
            stem_audio, _ = librosa.load(stem_path, sr=TARGET_SR, mono=True)
            stems[stem_name] = stem_audio

    if len(stems) == 0:
        return 0, 0

    # Load stem tonality info
    stem_tonality = None
    if dataset_type == "slakh":
        stem_tonality = load_slakh_metadata(track_path)
        # Check if we have at least one tonal stem
        tonal_stems = [s for s in stems.keys() if stem_tonality.get(s, True)]
        if not tonal_stems:
            # Skip tracks with only non-tonal instruments
            return 0, 0
    elif dataset_type == "musdb":
        # Use predefined MUSDB stem tonality
        stem_tonality = MUSDB_STEM_TONALITY

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
            # Skip very short chunks
            break

        # Get stem audio chunks
        stem_chunks = {
            name: audio[sample_offset:sample_offset + samples_per_chunk]
            for name, audio in stems.items()
        }

        # Create submixes (with tonality constraints for Slakh, filtering silent stems)
        submixes = create_submixes(
            stem_chunks, submixes_per_chunk, min_stems, max_stems,
            stem_tonality, stem_rms_threshold
        )

        # If no submixes created (all stems silent in this chunk), count as filtered
        if not submixes:
            examples_filtered += submixes_per_chunk
            chunk_idx += 1
            sample_offset += samples_per_chunk
            frame_offset += frames_per_chunk
            continue

        for submix_idx, (submix_audio, selected_stems) in enumerate(submixes):
            # Compute CQT features for submix (without log, applied on load)
            feature = compute_cqt_features(submix_audio)

            # Align feature frames to chord labels
            n_frames = feature.shape[1]
            if n_frames > len(chord_chunk):
                feature = feature[:, :len(chord_chunk)]
            elif n_frames < len(chord_chunk):
                chord_chunk = chord_chunk[:n_frames]

            # Create output filename
            output_filename = f"{track_id}_chunk{chunk_idx}_submix{submix_idx}.pt"
            output_path = output_dir / split / output_filename

            # Save example
            example = {
                'feature': feature,  # [144, T] - raw CQT magnitude
                'chord': chord_chunk.tolist(),  # [T] - chord IDs
                'metadata': {
                    'track_id': track_id,
                    'stems': selected_stems,
                    'chunk_idx': chunk_idx,
                    'submix_idx': submix_idx,
                    'dataset': dataset_type
                }
            }
            torch.save(example, output_path)
            examples_created += 1

        # Move to next chunk
        chunk_idx += 1
        sample_offset += samples_per_chunk
        frame_offset += frames_per_chunk

    return examples_created, examples_filtered


def process_track_wrapper(args):
    """Wrapper for multiprocessing that loads model per process."""
    (track_path, model_path, config_path, output_dir, split,
     chunk_duration, submixes_per_chunk, min_stems, max_stems, dataset_type, stem_rms_threshold) = args

    # Load model in this process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, config = load_btc_model(model_path, config_path)

    return process_track(
        track_path, model, mean, std, device, output_dir, split,
        chunk_duration, submixes_per_chunk, min_stems, max_stems, dataset_type, stem_rms_threshold
    )


def main():
    parser = argparse.ArgumentParser(description="Create BTC finetuning dataset from COCO Chorales, Slakh2100, and/or MUSDB18")
    parser.add_argument(
        "--coco_dir",
        type=str,
        default=os.path.expanduser("~/datasets/coco_chorales_contrastive/original"),
        help="Path to COCO Chorales directory"
    )
    parser.add_argument(
        "--slakh_dir",
        type=str,
        default=os.path.expanduser("~/datasets/slakh2100_contrastive/original"),
        help="Path to Slakh2100 directory"
    )
    parser.add_argument(
        "--musdb_dir",
        type=str,
        default=os.path.expanduser("~/datasets/musdb18"),
        help="Path to MUSDB18 directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["coco", "slakh", "musdb", "both", "all"],
        help="Which dataset(s) to process: coco, slakh, musdb, both (coco+slakh), or all"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/datasets/btc_finetuning"),
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
        default=3,
        help="Maximum number of stems in a submix"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (use 1 for GPU inference)"
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
        default=0.0,
        help="Filter out individual stems with RMS below this threshold per chunk (0.0 = no filtering, recommended: 0.03)"
    )
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    slakh_dir = Path(args.slakh_dir)
    musdb_dir = Path(args.musdb_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "valid").mkdir(parents=True, exist_ok=True)

    # Load model once for single-process mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, mean, std, config = load_btc_model(args.model_path, args.config_path)
    print("BTC model loaded successfully")

    # Process each split
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
            "datasets": args.dataset,
            "stem_rms_threshold": args.stem_rms_threshold,
        },
        "splits": {}
    }

    if args.stem_rms_threshold > 0:
        print(f"Stem-level silence filtering enabled: stems with RMS < {args.stem_rms_threshold} will be excluded per chunk")

    # Split mapping: Slakh's "validation" -> output "valid"
    # We skip Slakh's "test" split
    slakh_split_map = {"train": "train", "validation": "valid"}

    for output_split in ["train", "valid"]:
        # Collect tracks from selected dataset(s)
        track_info = []  # List of (track_path, dataset_type)

        if args.dataset in ["coco", "both", "all"]:
            coco_tracks = get_track_paths_coco(coco_dir, output_split)
            track_info.extend([(p, "coco") for p in coco_tracks])
            print(f"Found {len(coco_tracks)} COCO tracks for {output_split}")

        if args.dataset in ["slakh", "both", "all"]:
            # Map output split to Slakh split name
            slakh_split = "train" if output_split == "train" else "validation"
            slakh_tracks = get_track_paths_slakh(slakh_dir, slakh_split)
            track_info.extend([(p, "slakh") for p in slakh_tracks])
            print(f"Found {len(slakh_tracks)} Slakh tracks for {output_split}")

        if args.dataset in ["musdb", "all"]:
            # MUSDB uses "train" and "test" splits; map "valid" to "test"
            musdb_split = "train" if output_split == "train" else "test"
            musdb_tracks = get_track_paths_musdb(musdb_dir, musdb_split)
            track_info.extend([(p, "musdb") for p in musdb_tracks])
            print(f"Found {len(musdb_tracks)} MUSDB tracks for {output_split}")

        if not track_info:
            print(f"Warning: No tracks found for {output_split} split, skipping")
            continue

        if args.limit:
            track_info = track_info[:args.limit]

        print(f"\nProcessing {output_split} split: {len(track_info)} total tracks")

        total_examples = 0
        total_filtered = 0
        coco_count = sum(1 for _, dt in track_info if dt == "coco")
        slakh_count = sum(1 for _, dt in track_info if dt == "slakh")
        musdb_count = sum(1 for _, dt in track_info if dt == "musdb")

        if args.num_workers == 1:
            # Single process - use GPU
            for track_path, dataset_type in tqdm(track_info, desc=f"Processing {output_split}"):
                n_examples, n_filtered = process_track(
                    track_path, model, mean, std, device, output_dir, output_split,
                    args.chunk_duration, args.submixes_per_chunk,
                    args.min_stems, args.max_stems, dataset_type, args.stem_rms_threshold
                )
                total_examples += n_examples
                total_filtered += n_filtered
        else:
            # Multi-process - each process loads its own model
            tasks = [
                (track_path, args.model_path, args.config_path, output_dir, output_split,
                 args.chunk_duration, args.submixes_per_chunk,
                 args.min_stems, args.max_stems, dataset_type, args.stem_rms_threshold)
                for track_path, dataset_type in track_info
            ]

            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(process_track_wrapper, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {output_split}"):
                    n_examples, n_filtered = future.result()
                    total_examples += n_examples
                    total_filtered += n_filtered

        manifest["splits"][output_split] = {
            "num_tracks": len(track_info),
            "num_tracks_coco": coco_count,
            "num_tracks_slakh": slakh_count,
            "num_tracks_musdb": musdb_count,
            "num_examples": total_examples,
            "num_filtered": total_filtered
        }
        print(f"{output_split}: {total_examples} examples created, {total_filtered} filtered (silence) from {len(track_info)} tracks (COCO: {coco_count}, Slakh: {slakh_count}, MUSDB: {musdb_count})")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    print("\nDataset creation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
