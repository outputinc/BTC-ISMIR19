#!/usr/bin/env python
"""
Verify dataset completeness for BTC training.

Checks:
1. Audio/annotation file matching
2. File format validity
3. Reports missing files
4. Provides statistics

Usage:
    python verify_dataset.py --data_dir /data/music/chord_recognition
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


def find_mp3_path_isophonic(dirpath: Path, song_name: str) -> Optional[Path]:
    """Find matching MP3 file for an Isophonics annotation (from preprocess.py)."""
    for filename in dirpath.iterdir():
        if filename.suffix.lower() == ".mp3":
            last_dir = dirpath.name
            tmp = filename.stem.replace(last_dir, "")
            filename_lower = tmp.lower()
            filename_lower = " ".join(re.findall("[a-zA-Z]+", filename_lower))
            if song_name.lower().replace(" ", "") in filename_lower.replace(" ", ""):
                return filename
    return None


def find_mp3_path_robbiewilliams(dirpath: Path, song_name: str) -> Optional[Path]:
    """Find matching MP3 file for a Robbie Williams annotation (from preprocess.py)."""
    def song_pre(text):
        to_remove = ["'", '`', '(', ')', ' ', '&', 'and', 'And']
        for remove in to_remove:
            text = text.replace(remove, '')
        return text

    for filename in dirpath.iterdir():
        if filename.suffix.lower() == ".mp3":
            tmp = filename.stem.lower()
            tmp = tmp.replace("robbie williams", "")
            filename_lower = " ".join(re.findall("[a-zA-Z]+", tmp))
            filename_lower = song_pre(filename_lower)
            if song_pre(song_name.lower()).replace(" ", "") in filename_lower.replace(" ", ""):
                return filename
    return None


def uspop_pre(text: str) -> str:
    """Preprocess UsPop text (from preprocess.py)."""
    text = text.lower()
    text = text.replace('_', '')
    text = text.replace(' ', '')
    text = " ".join(re.findall("[a-zA-Z]+", text))
    return text


def verify_isophonics(data_dir: Path, verbose: bool = False) -> Dict:
    """Verify Isophonics dataset."""
    results = {
        "total_annotations": 0,
        "matched": 0,
        "missing_audio": [],
        "artists": defaultdict(lambda: {"annotations": 0, "matched": 0, "albums": set()}),
        "lab_files": [],
        "audio_files": []
    }

    isophonic_dir = data_dir / "isophonic"
    if not isophonic_dir.exists():
        print(f"Warning: Isophonics directory not found: {isophonic_dir}")
        return results

    # Walk through directory structure
    for dirpath, dirnames, filenames in os.walk(isophonic_dir):
        dirpath = Path(dirpath)
        if not dirnames:  # Leaf directory (album level)
            for filename in filenames:
                if filename.endswith(".lab"):
                    results["total_annotations"] += 1
                    results["lab_files"].append(dirpath / filename)

                    # Extract song name
                    tmp = filename.replace(".lab", "")
                    song_name = " ".join(re.findall("[a-zA-Z]+", tmp)).replace("CD", "")

                    # Find matching MP3
                    mp3_path = find_mp3_path_isophonic(dirpath, song_name)

                    # Determine artist from path
                    rel_path = dirpath.relative_to(isophonic_dir)
                    parts = rel_path.parts
                    artist = parts[0] if parts else "Unknown"

                    results["artists"][artist]["annotations"] += 1
                    if len(parts) > 1:
                        results["artists"][artist]["albums"].add(parts[1])

                    if mp3_path:
                        results["matched"] += 1
                        results["artists"][artist]["matched"] += 1
                        results["audio_files"].append(mp3_path)
                        if verbose:
                            print(f"  MATCH: {filename} -> {mp3_path.name}")
                    else:
                        results["missing_audio"].append({
                            "lab": str(dirpath / filename),
                            "expected_song": song_name,
                            "directory": str(dirpath)
                        })
                        if verbose:
                            print(f"  MISSING: {filename} (looking for: {song_name})")

    return results


def verify_uspop(data_dir: Path, verbose: bool = False) -> Dict:
    """Verify UsPop2002 dataset."""
    results = {
        "total_annotations": 0,
        "matched": 0,
        "missing_audio": [],
        "audio_in_dir": 0,
        "lab_files": [],
        "audio_files": []
    }

    uspop_dir = data_dir / "uspop"
    if not uspop_dir.exists():
        print(f"Warning: UsPop directory not found: {uspop_dir}")
        return results

    lab_dir = uspop_dir / "annotations" / "uspopLabels"
    audio_dir = uspop_dir / "audio"
    index_path = uspop_dir / "annotations" / "uspopLabels.txt"

    # Count audio files
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.MP3"))
        results["audio_in_dir"] = len(audio_files)

    # Load index if exists
    lab_list = []
    if index_path.exists():
        with open(index_path) as f:
            lab_list = [x.strip() for x in f.readlines() if x.strip()]

    # Process index entries
    for lab_entry in lab_list:
        # Parse lab path: ./uspopLabels/Artist/Album/01_Title.lab
        spl = lab_entry.split('/')
        if len(spl) >= 5:
            lab_artist = uspop_pre(spl[2])
            lab_title = uspop_pre(spl[4][3:-4])  # Remove track number and .lab

            lab_path = lab_entry.replace('./uspopLabels/', '')
            full_lab_path = lab_dir / lab_path

            if full_lab_path.exists():
                results["total_annotations"] += 1
                results["lab_files"].append(full_lab_path)

                # Look for matching audio
                found_match = False
                if audio_dir.exists():
                    for audio_file in audio_dir.iterdir():
                        if audio_file.suffix.lower() == ".mp3":
                            spl2 = audio_file.stem.split('-')
                            if len(spl2) >= 2:
                                mp3_artist = uspop_pre(spl2[0])
                                mp3_title = uspop_pre(spl2[1])

                                if lab_artist == mp3_artist and lab_title == mp3_title:
                                    results["matched"] += 1
                                    results["audio_files"].append(audio_file)
                                    found_match = True
                                    if verbose:
                                        print(f"  MATCH: {lab_path} -> {audio_file.name}")
                                    break

                if not found_match:
                    results["missing_audio"].append({
                        "lab": str(full_lab_path),
                        "expected_artist": lab_artist,
                        "expected_title": lab_title
                    })
                    if verbose:
                        print(f"  MISSING: {lab_artist} - {lab_title}")

    return results


def verify_robbiewilliams(data_dir: Path, verbose: bool = False) -> Dict:
    """Verify Robbie Williams dataset."""
    results = {
        "total_annotations": 0,
        "matched": 0,
        "missing_audio": [],
        "albums": defaultdict(lambda: {"annotations": 0, "matched": 0}),
        "lab_files": [],
        "audio_files": []
    }

    rw_dir = data_dir / "robbiewilliams"
    if not rw_dir.exists():
        print(f"Warning: Robbie Williams directory not found: {rw_dir}")
        return results

    chords_dir = rw_dir / "chords"
    audio_base_dir = rw_dir / "audio"

    if not chords_dir.exists():
        print(f"Warning: Robbie Williams chords directory not found: {chords_dir}")
        return results

    # Walk through chords directory
    for dirpath, dirnames, filenames in os.walk(chords_dir):
        dirpath = Path(dirpath)
        if not dirnames:  # Leaf directory (album level)
            for filename in filenames:
                if filename.endswith(".txt") and "README" not in filename:
                    results["total_annotations"] += 1
                    results["lab_files"].append(dirpath / filename)

                    # Extract song name
                    tmp = filename.replace(".txt", "")
                    song_name = " ".join(re.findall("[a-zA-Z]+", tmp)).replace("GTChords", "")

                    # Determine album from path
                    album = dirpath.name

                    results["albums"][album]["annotations"] += 1

                    # Look for audio in corresponding audio directory
                    audio_dir = audio_base_dir / album
                    mp3_path = None
                    if audio_dir.exists():
                        mp3_path = find_mp3_path_robbiewilliams(audio_dir, song_name)

                    if mp3_path:
                        results["matched"] += 1
                        results["albums"][album]["matched"] += 1
                        results["audio_files"].append(mp3_path)
                        if verbose:
                            print(f"  MATCH: {filename} -> {mp3_path.name}")
                    else:
                        results["missing_audio"].append({
                            "lab": str(dirpath / filename),
                            "expected_song": song_name,
                            "album": album,
                            "audio_dir": str(audio_dir) if audio_dir.exists() else "NOT FOUND"
                        })
                        if verbose:
                            print(f"  MISSING: {filename} (album: {album}, looking for: {song_name})")

    return results


def verify_audio_format(audio_path: Path, expected_sr: int = 22050) -> Dict:
    """Verify audio file can be loaded and check format."""
    if not LIBROSA_AVAILABLE:
        return {"status": "skipped", "reason": "librosa not available"}

    try:
        y, sr = librosa.load(audio_path, sr=None, duration=1.0)
        duration = librosa.get_duration(y=y, sr=sr)
        return {
            "status": "ok",
            "sample_rate": sr,
            "channels": 1 if y.ndim == 1 else y.shape[0],
            "duration_estimate": duration
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def print_dataset_summary(name: str, results: Dict):
    """Print summary for a dataset."""
    print(f"\n{'='*60}")
    print(f"{name} Dataset Summary")
    print('='*60)

    total = results.get("total_annotations", 0)
    matched = results.get("matched", 0)
    missing = total - matched

    print(f"Total annotations: {total}")
    print(f"Matched with audio: {matched}")
    print(f"Missing audio: {missing}")

    if total > 0:
        pct = (matched / total) * 100
        print(f"Match rate: {pct:.1f}%")

    # Print artist/album breakdown if available
    if "artists" in results and results["artists"]:
        print(f"\nBy Artist:")
        for artist, stats in sorted(results["artists"].items()):
            n_albums = len(stats.get("albums", set()))
            print(f"  {artist}: {stats['matched']}/{stats['annotations']} matched ({n_albums} albums)")

    if "albums" in results and results["albums"]:
        print(f"\nBy Album:")
        for album, stats in sorted(results["albums"].items()):
            print(f"  {album}: {stats['matched']}/{stats['annotations']} matched")

    if "audio_in_dir" in results:
        print(f"\nAudio files in directory: {results['audio_in_dir']}")

    # Show some missing files
    if results.get("missing_audio") and len(results["missing_audio"]) > 0:
        print(f"\nMissing audio examples (first 5):")
        for item in results["missing_audio"][:5]:
            if "expected_song" in item:
                print(f"  - {item.get('expected_song', 'unknown')} in {item.get('directory', item.get('album', 'unknown'))}")
            else:
                print(f"  - {item.get('expected_artist', '')} - {item.get('expected_title', '')}")


def write_missing_report(output_path: Path, all_results: Dict):
    """Write detailed report of missing files."""
    with open(output_path, 'w') as f:
        f.write("Missing Audio Files Report\n")
        f.write("="*60 + "\n\n")

        for dataset_name, results in all_results.items():
            missing = results.get("missing_audio", [])
            if missing:
                f.write(f"\n{dataset_name} ({len(missing)} missing):\n")
                f.write("-"*40 + "\n")
                for item in missing:
                    if "expected_song" in item:
                        f.write(f"  Song: {item['expected_song']}\n")
                        f.write(f"  Lab: {item['lab']}\n")
                        if 'album' in item:
                            f.write(f"  Album: {item['album']}\n")
                        if 'directory' in item:
                            f.write(f"  Directory: {item['directory']}\n")
                    else:
                        f.write(f"  Artist: {item.get('expected_artist', 'unknown')}\n")
                        f.write(f"  Title: {item.get('expected_title', 'unknown')}\n")
                        f.write(f"  Lab: {item['lab']}\n")
                    f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify BTC training dataset completeness"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/datasets/chord_recognition"),
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "isophonics", "uspop", "robbiewilliams"],
        help="Which dataset to verify"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual file matches"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Write missing files report to this path"
    )
    parser.add_argument(
        "--check_audio",
        action="store_true",
        help="Also verify audio files can be loaded (slower)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Verifying dataset at: {data_dir}")

    all_results = {}

    # Verify each dataset
    if args.dataset in ["all", "isophonics"]:
        print("\nVerifying Isophonics...")
        all_results["Isophonics"] = verify_isophonics(data_dir, args.verbose)
        print_dataset_summary("Isophonics", all_results["Isophonics"])

    if args.dataset in ["all", "uspop"]:
        print("\nVerifying UsPop2002...")
        all_results["UsPop2002"] = verify_uspop(data_dir, args.verbose)
        print_dataset_summary("UsPop2002", all_results["UsPop2002"])

    if args.dataset in ["all", "robbiewilliams"]:
        print("\nVerifying Robbie Williams...")
        all_results["RobbieWilliams"] = verify_robbiewilliams(data_dir, args.verbose)
        print_dataset_summary("Robbie Williams", all_results["RobbieWilliams"])

    # Check audio files if requested
    if args.check_audio:
        print("\n" + "="*60)
        print("Verifying Audio Files")
        print("="*60)

        for dataset_name, results in all_results.items():
            audio_files = results.get("audio_files", [])
            if audio_files:
                print(f"\n{dataset_name}: checking {len(audio_files)} files...")
                errors = []
                for af in audio_files[:10]:  # Check first 10
                    result = verify_audio_format(af)
                    if result["status"] == "error":
                        errors.append((af, result["error"]))
                        print(f"  ERROR: {af.name}: {result['error']}")
                    elif result["status"] == "ok":
                        print(f"  OK: {af.name} (sr={result['sample_rate']})")

                if len(audio_files) > 10:
                    print(f"  ... and {len(audio_files) - 10} more files")

    # Write missing files report
    if args.report:
        report_path = Path(args.report)
        write_missing_report(report_path, all_results)
        print(f"\nMissing files report written to: {report_path}")

    # Overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    total_annotations = sum(r.get("total_annotations", 0) for r in all_results.values())
    total_matched = sum(r.get("matched", 0) for r in all_results.values())
    total_missing = total_annotations - total_matched

    print(f"Total annotations: {total_annotations}")
    print(f"Total matched: {total_matched}")
    print(f"Total missing: {total_missing}")

    if total_annotations > 0:
        pct = (total_matched / total_annotations) * 100
        print(f"Overall match rate: {pct:.1f}%")

    if total_matched > 0:
        print("\nDataset is ready for processing!")
        print("Next step: python separate_audio.py --data_dir {}".format(data_dir))
    else:
        print("\nNo audio files found. Please add audio files before proceeding.")
        print("See download_annotations.py output for expected file locations.")

    return 0 if total_matched > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
