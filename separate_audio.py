#!/usr/bin/env python
"""
Batch source separation using Demucs for BTC training datasets.

Processes all audio files through Demucs htdemucs model to extract 4 stems:
- drums
- bass
- vocals
- other

Outputs are resampled to 22050 Hz for BTC compatibility.

Usage:
    python separate_audio.py --data_dir /data/music/chord_recognition
    python separate_audio.py --data_dir /data/music/chord_recognition --model htdemucs_ft

Requirements:
    pip install demucs
"""

import os
import sys
import argparse
import subprocess
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# Target sample rate for BTC
TARGET_SR = 22050

# Demucs stem names
STEM_NAMES = ["drums", "bass", "vocals", "other"]


def check_demucs_installed() -> bool:
    """Check if demucs is installed."""
    try:
        result = subprocess.run(
            ["demucs", "--help"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_audio_files_isophonics(data_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Get all audio files from Isophonics dataset.
    Returns list of (audio_path, relative_output_path) tuples.
    """
    files = []
    isophonic_dir = data_dir / "isophonic"

    if not isophonic_dir.exists():
        return files

    for dirpath, dirnames, filenames in os.walk(isophonic_dir):
        dirpath = Path(dirpath)
        for filename in filenames:
            if filename.lower().endswith(('.mp3', '.wav', '.flac')):
                audio_path = dirpath / filename
                # Preserve directory structure in output
                rel_path = audio_path.relative_to(isophonic_dir)
                # Output path: isophonic/Artist/Album/song_name/
                output_subdir = rel_path.parent / rel_path.stem
                files.append((audio_path, Path("isophonic") / output_subdir))

    return files


def get_audio_files_uspop(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Get all audio files from UsPop2002 dataset."""
    files = []
    audio_dir = data_dir / "uspop" / "audio"

    if not audio_dir.exists():
        return files

    for audio_path in audio_dir.iterdir():
        if audio_path.suffix.lower() in ['.mp3', '.wav', '.flac']:
            # Output path: uspop/song_name/
            output_subdir = Path("uspop") / audio_path.stem
            files.append((audio_path, output_subdir))

    return files


def get_audio_files_robbiewilliams(data_dir: Path) -> List[Tuple[Path, Path]]:
    """Get all audio files from Robbie Williams dataset."""
    files = []
    audio_dir = data_dir / "robbiewilliams" / "audio"

    if not audio_dir.exists():
        return files

    for dirpath, dirnames, filenames in os.walk(audio_dir):
        dirpath = Path(dirpath)
        for filename in filenames:
            if filename.lower().endswith(('.mp3', '.wav', '.flac')):
                audio_path = dirpath / filename
                # Preserve album structure
                rel_path = audio_path.relative_to(audio_dir)
                output_subdir = Path("robbiewilliams") / rel_path.parent / audio_path.stem
                files.append((audio_path, output_subdir))

    return files


def run_demucs(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
    device: str = "cuda",
    jobs: int = 0
) -> Tuple[bool, str]:
    """
    Run Demucs on a single audio file.

    Args:
        audio_path: Path to input audio
        output_dir: Base output directory
        model: Demucs model name
        device: cuda or cpu
        jobs: Number of parallel jobs (0 = auto)

    Returns:
        Tuple of (success, message)
    """
    cmd = [
        "demucs",
        "-n", model,
        "-d", device,
        "-o", str(output_dir),
        "--mp3",  # Keep original format for stems
    ]

    if jobs > 0:
        cmd.extend(["-j", str(jobs)])

    cmd.append(str(audio_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )

        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Demucs error: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Timeout (>10 minutes)"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def resample_stems(
    stems_dir: Path,
    target_sr: int = TARGET_SR
) -> Tuple[bool, str]:
    """
    Resample all stems in a directory to target sample rate.

    Args:
        stems_dir: Directory containing stem .wav files
        target_sr: Target sample rate

    Returns:
        Tuple of (success, message)
    """
    if not LIBROSA_AVAILABLE:
        return True, "Skipped (librosa not available)"

    try:
        for stem_name in STEM_NAMES:
            # Try different extensions
            stem_path = None
            for ext in ['.wav', '.mp3', '.flac']:
                candidate = stems_dir / f"{stem_name}{ext}"
                if candidate.exists():
                    stem_path = candidate
                    break

            if stem_path is None:
                continue

            # Load and resample
            y, sr = librosa.load(stem_path, sr=target_sr, mono=True)

            # Save as wav
            output_path = stems_dir / f"{stem_name}.wav"
            sf.write(output_path, y, target_sr)

            # Remove original if different
            if stem_path != output_path and stem_path.exists():
                stem_path.unlink()

        return True, "Resampled to 22050 Hz"

    except Exception as e:
        return False, f"Resample error: {str(e)}"


def process_single_file(
    audio_path: Path,
    output_subdir: Path,
    output_base_dir: Path,
    model: str,
    device: str,
    resample: bool
) -> Dict:
    """Process a single audio file through Demucs."""
    result = {
        "audio": str(audio_path),
        "output": str(output_base_dir / output_subdir),
        "success": False,
        "message": ""
    }

    # Check if already processed
    final_output_dir = output_base_dir / output_subdir
    if final_output_dir.exists():
        stems_exist = all(
            (final_output_dir / f"{stem}.wav").exists()
            for stem in STEM_NAMES
        )
        if stems_exist:
            result["success"] = True
            result["message"] = "Already processed"
            return result

    # Run Demucs
    success, message = run_demucs(audio_path, output_base_dir, model, device)

    if not success:
        result["message"] = message
        return result

    # Demucs outputs to: output_dir/model/audio_stem/
    demucs_output = output_base_dir / model / audio_path.stem

    if not demucs_output.exists():
        result["message"] = f"Demucs output not found at {demucs_output}"
        return result

    # Move/rename to final location
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)

    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)

    shutil.move(str(demucs_output), str(final_output_dir))

    # Resample if requested
    if resample:
        success, msg = resample_stems(final_output_dir)
        if not success:
            result["message"] = msg
            return result

    result["success"] = True
    result["message"] = "Processed successfully"
    return result


def cleanup_demucs_dirs(output_dir: Path, model: str):
    """Clean up intermediate Demucs directories."""
    model_dir = output_dir / model
    if model_dir.exists() and model_dir.is_dir():
        # Check if empty
        remaining = list(model_dir.iterdir())
        if not remaining:
            model_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(
        description="Batch source separation using Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all datasets
    python separate_audio.py --data_dir /data/music/chord_recognition

    # Process specific dataset
    python separate_audio.py --data_dir /data/music/chord_recognition --dataset isophonics

    # Use CPU instead of GPU
    python separate_audio.py --data_dir /data/music/chord_recognition --device cpu

    # Use fine-tuned model
    python separate_audio.py --data_dir /data/music/chord_recognition --model htdemucs_ft
        """
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/datasets/chord_recognition"),
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for separated stems (default: data_dir/separated)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "isophonics", "uspop", "robbiewilliams"],
        help="Which dataset to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"],
        help="Demucs model to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run Demucs on"
    )
    parser.add_argument(
        "--no_resample",
        action="store_true",
        help="Skip resampling to 22050 Hz"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (use 1 for GPU)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already processed)"
    )
    args = parser.parse_args()

    # Check Demucs
    if not check_demucs_installed():
        print("Error: Demucs is not installed.")
        print("Install with: pip install demucs")
        return 1

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "separated"

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    # Collect all audio files
    all_files = []

    if args.dataset in ["all", "isophonics"]:
        files = get_audio_files_isophonics(data_dir)
        print(f"Found {len(files)} Isophonics audio files")
        all_files.extend(files)

    if args.dataset in ["all", "uspop"]:
        files = get_audio_files_uspop(data_dir)
        print(f"Found {len(files)} UsPop2002 audio files")
        all_files.extend(files)

    if args.dataset in ["all", "robbiewilliams"]:
        files = get_audio_files_robbiewilliams(data_dir)
        print(f"Found {len(files)} Robbie Williams audio files")
        all_files.extend(files)

    if not all_files:
        print("No audio files found!")
        return 1

    # Limit if requested
    if args.limit:
        all_files = all_files[:args.limit]

    print(f"\nTotal files to process: {len(all_files)}")

    # Filter already processed if resuming
    if args.resume:
        original_count = len(all_files)
        filtered_files = []
        for audio_path, output_subdir in all_files:
            final_dir = output_dir / output_subdir
            if not final_dir.exists() or not all(
                (final_dir / f"{stem}.wav").exists() for stem in STEM_NAMES
            ):
                filtered_files.append((audio_path, output_subdir))
        all_files = filtered_files
        print(f"Resuming: {original_count - len(all_files)} already processed, {len(all_files)} remaining")

    if not all_files:
        print("All files already processed!")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "errors": []
    }

    resample = not args.no_resample and LIBROSA_AVAILABLE

    if TQDM_AVAILABLE:
        iterator = tqdm.tqdm(all_files, desc="Processing")
    else:
        iterator = all_files

    for i, (audio_path, output_subdir) in enumerate(iterator):
        if not TQDM_AVAILABLE:
            print(f"Processing [{i+1}/{len(all_files)}]: {audio_path.name}")

        result = process_single_file(
            audio_path,
            output_subdir,
            output_dir,
            args.model,
            args.device,
            resample
        )

        if result["success"]:
            if "Already" in result["message"]:
                results["skipped"] += 1
            else:
                results["success"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(result)
            print(f"  ERROR: {result['message']}")

    # Cleanup intermediate directories
    cleanup_demucs_dirs(output_dir, args.model)

    # Print summary
    print("\n" + "="*60)
    print("SEPARATION SUMMARY")
    print("="*60)
    print(f"Successfully processed: {results['success']}")
    print(f"Skipped (already done): {results['skipped']}")
    print(f"Failed: {results['failed']}")

    if results["errors"]:
        print(f"\nFailed files:")
        for err in results["errors"][:10]:
            print(f"  - {err['audio']}: {err['message']}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    # Save results log
    log_path = output_dir / "separation_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            "summary": {
                "total": len(all_files),
                "success": results["success"],
                "skipped": results["skipped"],
                "failed": results["failed"]
            },
            "errors": results["errors"],
            "config": {
                "model": args.model,
                "device": args.device,
                "resample": resample
            }
        }, f, indent=2)
    print(f"\nLog saved to: {log_path}")

    # Print next steps
    if results["success"] + results["skipped"] > 0:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"""
Source separation complete! To create the finetuning dataset:

    python create_finetuning_dataset_labeled.py \\
        --data_dir {data_dir} \\
        --separated_dir {output_dir} \\
        --output_dir ~/datasets/btc_finetuning_labeled

Then finetune the model:

    python finetune_btc.py \\
        --data_dir ~/datasets/btc_finetuning_labeled \\
        --epochs 20
""")

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
