#!/usr/bin/env python
"""
Create finetuning dataset for BTC chord recognition using original human-annotated labels.

This script differs from create_finetuning_dataset.py by using the original .lab chord
annotations instead of BTC inference labels. This is useful when you have access to
ground-truth annotations (Isophonics, UsPop2002, Robbie Williams datasets).

Pipeline:
1. Load original chord annotations from .lab files
2. Align chord labels to audio frames
3. Load source-separated stems from Demucs output
4. Create N random submixes per audio chunk
5. Save CQT features with original chord labels

Usage:
    python create_finetuning_dataset_labeled.py \
        --data_dir /data/music/chord_recognition \
        --separated_dir /data/music/chord_recognition/separated \
        --output_dir ~/datasets/btc_finetuning_labeled

Requirements:
    - Original annotations (.lab files) in data_dir
    - Source-separated stems in separated_dir (from separate_audio.py)
"""

import os
import sys
import json
import random
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.chords import Chords


# Constants from BTC config (matching create_finetuning_dataset.py)
TARGET_SR = 22050
N_BINS = 144
BINS_PER_OCTAVE = 24
HOP_LENGTH = 2048
TIMESTEP = 108  # frames per chunk
CHUNK_DURATION = 10.0  # seconds
SKIP_INTERVAL = 5.0  # seconds between chunk starts
NUM_CHORDS = 25  # major/minor vocabulary (0-23 = chords, 24 = N)

# Time interval per frame
TIME_INTERVAL = HOP_LENGTH / TARGET_SR  # ~0.093 seconds

# Stem names from Demucs
STEM_NAMES = ["drums", "bass", "vocals", "other"]

# Tonal stems (for ensuring chord context)
TONAL_STEMS = {"bass", "vocals", "other"}


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


def load_chord_annotations(lab_path: Path, chord_class: Chords) -> pd.DataFrame:
    """
    Load chord annotations from a .lab file.

    Uses the same parsing as utils/chords.py and utils/preprocess.py.

    Args:
        lab_path: Path to .lab annotation file
        chord_class: Chords instance for parsing

    Returns:
        DataFrame with columns: start, end, chord_id
    """
    return chord_class.get_converted_chord(str(lab_path))


def align_chords_to_frames(
    chord_df: pd.DataFrame,
    start_time: float,
    duration: float,
    time_interval: float = TIME_INTERVAL,
    shift_factor: int = 0
) -> List[int]:
    """
    Align chord annotations to frame-level labels.

    Adapted from utils/preprocess.py:217-252.

    Args:
        chord_df: DataFrame with start, end, chord_id columns
        start_time: Start time in seconds
        duration: Duration in seconds
        time_interval: Time per frame
        shift_factor: Pitch shift in semitones (for augmentation)

    Returns:
        List of chord IDs for each frame
    """
    chord_list = []
    cur_sec = start_time
    end_sec = start_time + duration

    while cur_sec < end_sec:
        try:
            # Find chords that overlap with current time window
            available_chords = chord_df.loc[
                (chord_df['start'] <= cur_sec) &
                (chord_df['end'] > cur_sec + time_interval)
            ].copy()

            if len(available_chords) == 0:
                # Try partial overlap
                available_chords = chord_df.loc[
                    ((chord_df['start'] >= cur_sec) & (chord_df['start'] <= cur_sec + time_interval)) |
                    ((chord_df['end'] >= cur_sec) & (chord_df['end'] <= cur_sec + time_interval))
                ].copy()

            if len(available_chords) == 1:
                chord = available_chords['chord_id'].iloc[0]
            elif len(available_chords) > 1:
                # Choose chord with longest overlap
                max_starts = available_chords.apply(lambda row: max(row['start'], cur_sec), axis=1)
                available_chords['max_start'] = max_starts
                min_ends = available_chords.apply(lambda row: min(row['end'], cur_sec + time_interval), axis=1)
                available_chords['min_end'] = min_ends
                chord_lengths = available_chords['min_end'] - available_chords['max_start']
                available_chords['chord_length'] = chord_lengths
                chord = available_chords.loc[available_chords['chord_length'].idxmax()]['chord_id']
            else:
                chord = 24  # No chord

        except Exception:
            chord = 24  # No chord on error

        # Apply pitch shift to chord ID
        if chord != 24 and shift_factor != 0:
            chord = int(chord) + shift_factor * 2
            chord = chord % 24

        chord_list.append(int(chord))
        cur_sec += time_interval

    return chord_list


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


def create_submixes(
    stems: Dict[str, np.ndarray],
    n_submixes: int,
    min_stems: int = 1,
    max_stems: int = 3,
    ensure_tonal: bool = True,
    rms_threshold: float = 0.03
) -> List[Tuple[np.ndarray, List[str]]]:
    """
    Generate random stem combinations for submixes.

    Ensures at least one tonal stem (bass/vocals/other) is included
    to maintain harmonic context for chord recognition.

    Args:
        stems: Dictionary of stem_name -> audio array
        n_submixes: Number of submixes to create
        min_stems: Minimum stems per submix
        max_stems: Maximum stems per submix
        ensure_tonal: Require at least one tonal stem
        rms_threshold: Filter silent stems

    Returns:
        List of (submix_audio, stem_names) tuples
    """
    # Filter silent stems
    if rms_threshold > 0:
        stems = filter_active_stems(stems, rms_threshold)

    if not stems:
        return []

    stem_names = list(stems.keys())
    submixes = []

    # Separate tonal and non-tonal stems
    tonal = [s for s in stem_names if s in TONAL_STEMS]
    non_tonal = [s for s in stem_names if s not in TONAL_STEMS]

    # Skip if no tonal stems and we require them
    if ensure_tonal and not tonal:
        return []

    for _ in range(n_submixes):
        k = random.randint(min_stems, min(max_stems, len(stem_names)))

        if ensure_tonal and tonal:
            # Pick at least one tonal stem first
            selected = [random.choice(tonal)]
            remaining = [s for s in stem_names if s not in selected]

            if k > 1 and remaining:
                additional = random.sample(remaining, min(k - 1, len(remaining)))
                selected.extend(additional)
        else:
            selected = random.sample(stem_names, min(k, len(stem_names)))

        # Sum selected stems
        submix_audio = np.zeros_like(next(iter(stems.values())))
        for name in selected:
            submix_audio += stems[name]

        submixes.append((submix_audio, selected))

    return submixes


def find_lab_path_isophonic(audio_path: Path) -> Optional[Path]:
    """Find matching .lab file for an Isophonics audio file."""
    # .lab file is in the same directory as audio
    directory = audio_path.parent

    # Try direct name match
    lab_path = directory / (audio_path.stem + ".lab")
    if lab_path.exists():
        return lab_path

    # Try fuzzy match
    song_name = " ".join(re.findall("[a-zA-Z]+", audio_path.stem)).replace("CD", "")

    for lab_file in directory.glob("*.lab"):
        tmp = lab_file.stem
        lab_name = " ".join(re.findall("[a-zA-Z]+", tmp)).replace("CD", "")
        if song_name.lower().replace(" ", "") in lab_name.lower().replace(" ", ""):
            return lab_file

    return None


def find_lab_path_uspop(
    audio_path: Path,
    annotations_dir: Path,
    index_entries: List[str]
) -> Optional[Path]:
    """Find matching .lab file for a UsPop2002 audio file."""
    def uspop_pre(text):
        text = text.lower().replace('_', '').replace(' ', '')
        return " ".join(re.findall("[a-zA-Z]+", text))

    spl = audio_path.stem.split('-')
    if len(spl) < 2:
        return None

    mp3_artist = uspop_pre(spl[0])
    mp3_title = uspop_pre(spl[1])

    for entry in index_entries:
        spl2 = entry.split('/')
        if len(spl2) >= 5:
            lab_artist = uspop_pre(spl2[2])
            lab_title = uspop_pre(spl2[4][3:-4])

            if lab_artist == mp3_artist and lab_title == mp3_title:
                lab_rel = entry.replace('./uspopLabels/', '')
                lab_path = annotations_dir / lab_rel
                if lab_path.exists():
                    return lab_path

    return None


def find_lab_path_robbiewilliams(audio_path: Path, chords_dir: Path) -> Optional[Path]:
    """Find matching .txt chord file for a Robbie Williams audio file."""
    def song_pre(text):
        for remove in ["'", '`', '(', ')', ' ', '&', 'and', 'And']:
            text = text.replace(remove, '')
        return text

    # Get album from path
    album = audio_path.parent.name

    song_name = audio_path.stem.lower()
    song_name = song_name.replace("robbie williams", "")
    song_name = " ".join(re.findall("[a-zA-Z]+", song_name))
    song_name = song_pre(song_name)

    # Look in corresponding chords directory
    album_chords = chords_dir / album
    if not album_chords.exists():
        return None

    for txt_file in album_chords.glob("*.txt"):
        if "README" in txt_file.name:
            continue

        tmp = txt_file.stem
        lab_name = " ".join(re.findall("[a-zA-Z]+", tmp)).replace("GTChords", "")
        lab_name = song_pre(lab_name)

        if song_name.replace(" ", "") in lab_name.replace(" ", ""):
            return txt_file

    return None


def get_track_info(
    data_dir: Path,
    separated_dir: Path
) -> List[Dict]:
    """
    Get all tracks with their audio, stems, and annotation paths.

    Returns list of dicts with keys:
        - track_id: Unique identifier
        - dataset: isophonics, uspop, or robbiewilliams
        - audio_path: Path to original audio (for verification)
        - stems_dir: Path to separated stems
        - lab_path: Path to chord annotations
    """
    tracks = []
    chord_class = Chords()

    # Isophonics
    iso_sep_dir = separated_dir / "isophonic"
    if iso_sep_dir.exists():
        for stems_dir in iso_sep_dir.rglob("*"):
            if stems_dir.is_dir() and (stems_dir / "bass.wav").exists():
                # Find corresponding lab file
                # stems_dir structure: separated/isophonic/Artist/Album/SongName/
                rel_parts = stems_dir.relative_to(iso_sep_dir).parts
                if len(rel_parts) >= 3:
                    artist, album, song = rel_parts[0], rel_parts[1], rel_parts[2]
                    original_dir = data_dir / "isophonic" / artist / album

                    if original_dir.exists():
                        # Find lab file
                        for lab_file in original_dir.glob("*.lab"):
                            lab_song = " ".join(re.findall("[a-zA-Z]+", lab_file.stem)).replace("CD", "")
                            stem_song = " ".join(re.findall("[a-zA-Z]+", song)).replace("CD", "")
                            if lab_song.lower().replace(" ", "") == stem_song.lower().replace(" ", "") or \
                               stem_song.lower().replace(" ", "") in lab_song.lower().replace(" ", ""):
                                tracks.append({
                                    "track_id": f"iso_{artist}_{album}_{song}",
                                    "dataset": "isophonics",
                                    "stems_dir": stems_dir,
                                    "lab_path": lab_file
                                })
                                break

    # UsPop2002
    uspop_sep_dir = separated_dir / "uspop"
    uspop_ann_dir = data_dir / "uspop" / "annotations" / "uspopLabels"
    uspop_index = data_dir / "uspop" / "annotations" / "uspopLabels.txt"

    if uspop_sep_dir.exists() and uspop_ann_dir.exists():
        # Load index
        index_entries = []
        if uspop_index.exists():
            with open(uspop_index) as f:
                index_entries = [x.strip() for x in f.readlines() if x.strip()]

        for stems_dir in uspop_sep_dir.iterdir():
            if stems_dir.is_dir() and (stems_dir / "bass.wav").exists():
                # stems_dir name is the song name
                song_name = stems_dir.name

                # Create dummy audio path for matching
                dummy_audio = data_dir / "uspop" / "audio" / (song_name + ".mp3")
                lab_path = find_lab_path_uspop(dummy_audio, uspop_ann_dir, index_entries)

                if lab_path:
                    tracks.append({
                        "track_id": f"uspop_{song_name}",
                        "dataset": "uspop",
                        "stems_dir": stems_dir,
                        "lab_path": lab_path
                    })

    # Robbie Williams
    rw_sep_dir = separated_dir / "robbiewilliams"
    rw_chords_dir = data_dir / "robbiewilliams" / "chords"

    if rw_sep_dir.exists() and rw_chords_dir.exists():
        for album_dir in rw_sep_dir.iterdir():
            if album_dir.is_dir():
                for stems_dir in album_dir.iterdir():
                    if stems_dir.is_dir() and (stems_dir / "bass.wav").exists():
                        # Create dummy audio path for matching
                        dummy_audio = data_dir / "robbiewilliams" / "audio" / album_dir.name / (stems_dir.name + ".mp3")
                        lab_path = find_lab_path_robbiewilliams(dummy_audio, rw_chords_dir)

                        if lab_path:
                            tracks.append({
                                "track_id": f"rw_{album_dir.name}_{stems_dir.name}",
                                "dataset": "robbiewilliams",
                                "stems_dir": stems_dir,
                                "lab_path": lab_path
                            })

    return tracks


def process_track(
    track_info: Dict,
    output_dir: Path,
    split: str,
    chord_class: Chords,
    chunk_duration: float = CHUNK_DURATION,
    skip_interval: float = SKIP_INTERVAL,
    submixes_per_chunk: int = 3,
    min_stems: int = 1,
    max_stems: int = 3,
    rms_threshold: float = 0.03,
    pitch_shifts: List[int] = None
) -> Tuple[int, int]:
    """
    Process a single track: create submixes with original chord labels.

    Args:
        track_info: Dict with track_id, dataset, stems_dir, lab_path
        output_dir: Output directory
        split: train or valid
        chord_class: Chords instance
        chunk_duration: Duration of each chunk
        skip_interval: Seconds between chunk starts
        submixes_per_chunk: Number of submixes per chunk
        min_stems: Minimum stems per submix
        max_stems: Maximum stems per submix
        rms_threshold: Filter silent stems
        pitch_shifts: List of pitch shift values for augmentation

    Returns:
        Tuple of (examples_created, examples_filtered)
    """
    if pitch_shifts is None:
        pitch_shifts = [0]  # No augmentation by default

    track_id = track_info["track_id"]
    stems_dir = track_info["stems_dir"]
    lab_path = track_info["lab_path"]

    examples_created = 0
    examples_filtered = 0

    # Load chord annotations
    try:
        chord_df = load_chord_annotations(lab_path, chord_class)
    except Exception as e:
        print(f"Error loading annotations for {track_id}: {e}")
        return 0, 0

    # Load stems
    stems = {}
    for stem_name in STEM_NAMES:
        stem_path = stems_dir / f"{stem_name}.wav"
        if stem_path.exists():
            audio, _ = librosa.load(stem_path, sr=TARGET_SR, mono=True)
            stems[stem_name] = audio

    if not stems:
        return 0, 0

    # Get common length
    min_length = min(len(s) for s in stems.values())
    stems = {k: v[:min_length] for k, v in stems.items()}

    # Get annotation duration
    ann_duration = chord_df['end'].max()
    audio_duration = min_length / TARGET_SR

    # Use the shorter of annotation or audio duration
    max_duration = min(ann_duration, audio_duration)

    # Process chunks
    samples_per_chunk = int(chunk_duration * TARGET_SR)
    frames_per_chunk = int(chunk_duration * TARGET_SR / HOP_LENGTH)

    for shift in pitch_shifts:
        current_start = 0.0

        while current_start + chunk_duration < max_duration:
            # Get chord labels for this chunk
            chord_list = align_chords_to_frames(
                chord_df,
                current_start,
                chunk_duration,
                TIME_INTERVAL,
                shift_factor=shift
            )

            # Truncate to expected length
            if len(chord_list) > frames_per_chunk:
                chord_list = chord_list[:frames_per_chunk]

            # Get stem audio chunks
            start_sample = int(current_start * TARGET_SR)
            end_sample = start_sample + samples_per_chunk

            stem_chunks = {
                name: audio[start_sample:end_sample]
                for name, audio in stems.items()
            }

            # Create submixes
            submixes = create_submixes(
                stem_chunks,
                submixes_per_chunk,
                min_stems,
                max_stems,
                ensure_tonal=True,
                rms_threshold=rms_threshold
            )

            if not submixes:
                examples_filtered += submixes_per_chunk
                current_start += skip_interval
                continue

            chunk_idx = int(current_start / skip_interval)

            for submix_idx, (submix_audio, selected_stems) in enumerate(submixes):
                # Pitch shift audio if needed
                if shift != 0:
                    try:
                        submix_audio = librosa.effects.pitch_shift(
                            submix_audio,
                            sr=TARGET_SR,
                            n_steps=shift
                        )
                    except Exception:
                        continue

                # Compute CQT features
                feature = compute_cqt_features(submix_audio)

                # Align frames
                n_frames = feature.shape[1]
                if n_frames > len(chord_list):
                    feature = feature[:, :len(chord_list)]
                elif n_frames < len(chord_list):
                    chord_list = chord_list[:n_frames]

                # Create filename
                shift_str = f"_shift{shift}" if shift != 0 else ""
                filename = f"{track_id}_chunk{chunk_idx}_submix{submix_idx}{shift_str}.pt"
                output_path = output_dir / split / filename

                # Save example
                example = {
                    'feature': feature,  # [144, T] - raw CQT magnitude
                    'chord': chord_list,  # [T] - chord IDs
                    'metadata': {
                        'track_id': track_id,
                        'dataset': track_info["dataset"],
                        'stems': selected_stems,
                        'chunk_idx': chunk_idx,
                        'submix_idx': submix_idx,
                        'pitch_shift': shift,
                        'start_time': current_start,
                        'lab_source': str(lab_path.name)
                    }
                }
                torch.save(example, output_path)
                examples_created += 1

            current_start += skip_interval

    return examples_created, examples_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Create BTC finetuning dataset using original chord annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates a finetuning dataset using original human-annotated chord labels
instead of BTC inference. Use this when you have access to ground-truth annotations.

Prerequisites:
1. Download annotations: python download_annotations.py --output_dir DATA_DIR
2. Add audio files to appropriate directories
3. Run source separation: python separate_audio.py --data_dir DATA_DIR

Example:
    python create_finetuning_dataset_labeled.py \\
        --data_dir /data/music/chord_recognition \\
        --separated_dir /data/music/chord_recognition/separated \\
        --output_dir ~/datasets/btc_finetuning_labeled \\
        --submixes_per_chunk 3
        """
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/datasets/chord_recognition"),
        help="Path to dataset with annotations"
    )
    parser.add_argument(
        "--separated_dir",
        type=str,
        default=None,
        help="Path to separated stems (default: data_dir/separated)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/datasets/btc_finetuning_labeled"),
        help="Output directory for finetuning dataset"
    )
    parser.add_argument(
        "--chunk_duration",
        type=float,
        default=CHUNK_DURATION,
        help="Duration of each chunk in seconds"
    )
    parser.add_argument(
        "--skip_interval",
        type=float,
        default=SKIP_INTERVAL,
        help="Seconds between chunk starts"
    )
    parser.add_argument(
        "--submixes_per_chunk",
        type=int,
        default=3,
        help="Number of submixes per chunk"
    )
    parser.add_argument(
        "--min_stems",
        type=int,
        default=1,
        help="Minimum stems per submix"
    )
    parser.add_argument(
        "--max_stems",
        type=int,
        default=3,
        help="Maximum stems per submix"
    )
    parser.add_argument(
        "--rms_threshold",
        type=float,
        default=0.03,
        help="RMS threshold for filtering silent stems"
    )
    parser.add_argument(
        "--valid_split",
        type=float,
        default=0.1,
        help="Fraction of tracks to use for validation"
    )
    parser.add_argument(
        "--pitch_augment",
        action="store_true",
        help="Enable pitch shift augmentation (-5 to +6 semitones)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tracks (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    separated_dir = Path(args.separated_dir) if args.separated_dir else data_dir / "separated"
    output_dir = Path(args.output_dir)

    print(f"Data directory: {data_dir}")
    print(f"Separated stems: {separated_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directories
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "valid").mkdir(parents=True, exist_ok=True)

    # Initialize chord parser
    chord_class = Chords()

    # Get all tracks
    print("\nScanning for tracks...")
    tracks = get_track_info(data_dir, separated_dir)

    if not tracks:
        print("No tracks found! Ensure:")
        print("  1. Annotations exist in data_dir")
        print("  2. Separated stems exist in separated_dir")
        print("  3. File naming matches between annotations and stems")
        return 1

    print(f"Found {len(tracks)} tracks with matched annotations and stems")

    # Count by dataset
    by_dataset = defaultdict(list)
    for t in tracks:
        by_dataset[t["dataset"]].append(t)

    for ds, ts in by_dataset.items():
        print(f"  {ds}: {len(ts)} tracks")

    # Limit if requested
    if args.limit:
        tracks = tracks[:args.limit]
        print(f"Limited to {len(tracks)} tracks")

    # Split into train/valid
    random.shuffle(tracks)
    n_valid = max(1, int(len(tracks) * args.valid_split))
    valid_tracks = tracks[:n_valid]
    train_tracks = tracks[n_valid:]

    print(f"\nTrain: {len(train_tracks)} tracks")
    print(f"Valid: {len(valid_tracks)} tracks")

    # Pitch shifts for augmentation
    pitch_shifts = [0]
    if args.pitch_augment:
        pitch_shifts = list(range(-5, 7))  # -5 to +6 semitones
        print(f"Pitch augmentation enabled: {pitch_shifts}")

    # Process tracks
    manifest = {
        "config": {
            "chunk_duration": args.chunk_duration,
            "skip_interval": args.skip_interval,
            "submixes_per_chunk": args.submixes_per_chunk,
            "min_stems": args.min_stems,
            "max_stems": args.max_stems,
            "target_sr": TARGET_SR,
            "n_bins": N_BINS,
            "hop_length": HOP_LENGTH,
            "num_chords": NUM_CHORDS,
            "rms_threshold": args.rms_threshold,
            "pitch_augment": args.pitch_augment,
            "pitch_shifts": pitch_shifts,
            "label_source": "original_annotations"
        },
        "splits": {}
    }

    for split_name, split_tracks in [("train", train_tracks), ("valid", valid_tracks)]:
        print(f"\nProcessing {split_name} split...")

        total_created = 0
        total_filtered = 0
        dataset_counts = defaultdict(int)

        for track in tqdm(split_tracks, desc=split_name):
            created, filtered = process_track(
                track,
                output_dir,
                split_name,
                chord_class,
                chunk_duration=args.chunk_duration,
                skip_interval=args.skip_interval,
                submixes_per_chunk=args.submixes_per_chunk,
                min_stems=args.min_stems,
                max_stems=args.max_stems,
                rms_threshold=args.rms_threshold,
                pitch_shifts=pitch_shifts
            )
            total_created += created
            total_filtered += filtered
            dataset_counts[track["dataset"]] += created

        manifest["splits"][split_name] = {
            "num_tracks": len(split_tracks),
            "num_examples": total_created,
            "num_filtered": total_filtered,
            "by_dataset": dict(dataset_counts)
        }

        print(f"{split_name}: {total_created} examples created, {total_filtered} filtered")
        for ds, count in dataset_counts.items():
            print(f"  {ds}: {count} examples")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    # Print next steps
    print("\n" + "="*60)
    print("DATASET CREATION COMPLETE")
    print("="*60)
    print(f"""
Dataset created at: {output_dir}

Train examples: {manifest['splits']['train']['num_examples']}
Valid examples: {manifest['splits']['valid']['num_examples']}

To finetune the model:

    python finetune_btc.py \\
        --data_dir {output_dir} \\
        --model_path ./test/btc_model.pt \\
        --epochs 20 \\
        --batch_size 32 \\
        --learning_rate 1e-5
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
