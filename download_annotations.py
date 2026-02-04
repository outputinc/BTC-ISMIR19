#!/usr/bin/env python
"""
Download and organize chord annotations for BTC training datasets.

Supported datasets:
- Isophonics: Beatles, Queen, Carole King, Zweieck, Michael Jackson
- UsPop2002: 195 pop songs (annotations from GitHub)
- Robbie Williams: 65 songs (annotations from POLIMI)

Usage:
    python download_annotations.py --output_dir /data/music/chord_recognition

Note: Audio files must be obtained separately due to copyright restrictions.
"""

import os
import sys
import argparse
import zipfile
import tarfile
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import subprocess


def download_file(url: str, dest_path: Path, desc: str = None) -> bool:
    """Download a file with progress indication."""
    if desc:
        print(f"Downloading {desc}...")
    try:
        urlretrieve(url, dest_path)
        return True
    except URLError as e:
        print(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error downloading {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


def extract_tar(tar_path: Path, extract_to: Path):
    """Extract a tar/tar.gz file."""
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:*') as tf:
        tf.extractall(extract_to)


def git_clone(repo_url: str, dest_path: Path) -> bool:
    """Clone a git repository."""
    print(f"Cloning {repo_url}...")
    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, str(dest_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Git clone failed: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        print("Error: git is not installed")
        return False


def download_isophonics(output_dir: Path) -> dict:
    """
    Download Isophonics annotations.

    Note: Isophonics annotations are available at http://isophonics.net/datasets
    Direct download links (as of 2024):
    - Beatles: http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz
    - Queen: http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz
    - Carole King: http://isophonics.net/files/annotations/Carole%20King%20Annotations.tar.gz
    - Zweieck: http://isophonics.net/files/annotations/Zweieck%20Annotations.tar.gz
    - Michael Jackson: http://isophonics.net/files/annotations/Michael%20Jackson%20Annotations.tar.gz
    """
    isophonics_dir = output_dir / "isophonic"
    isophonics_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }

    # Isophonics annotation URLs
    annotation_urls = {
        "Beatles": "http://isophonics.net/files/annotations/The%20Beatles%20Annotations.tar.gz",
        "Queen": "http://isophonics.net/files/annotations/Queen%20Annotations.tar.gz",
        "Carole_King": "http://isophonics.net/files/annotations/Carole%20King%20Annotations.tar.gz",
        "Zweieck": "http://isophonics.net/files/annotations/Zweieck%20Annotations.tar.gz",
        "Michael_Jackson": "http://isophonics.net/files/annotations/Michael%20Jackson%20Annotations.tar.gz",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for artist, url in annotation_urls.items():
            # Check if already exists
            artist_dir = isophonics_dir / artist
            if artist_dir.exists() and any(artist_dir.rglob("*.lab")):
                print(f"Skipping {artist} - already exists")
                results["skipped"].append(artist)
                continue

            tar_path = tmpdir / f"{artist}.tar.gz"
            if download_file(url, tar_path, f"{artist} annotations"):
                try:
                    extract_to = tmpdir / artist
                    extract_to.mkdir(exist_ok=True)
                    extract_tar(tar_path, extract_to)

                    # Find extracted content (usually in a subdirectory)
                    extracted_dirs = list(extract_to.iterdir())
                    if extracted_dirs:
                        source = extracted_dirs[0]
                        # Move to final location
                        if artist_dir.exists():
                            shutil.rmtree(artist_dir)
                        shutil.move(str(source), str(artist_dir))
                        results["success"].append(artist)
                        print(f"  Installed {artist} annotations")
                    else:
                        results["failed"].append(artist)
                except Exception as e:
                    print(f"  Error extracting {artist}: {e}")
                    results["failed"].append(artist)
            else:
                results["failed"].append(artist)

    return results


def download_uspop(output_dir: Path) -> dict:
    """
    Download UsPop2002 annotations from GitHub.

    Repository: https://github.com/tmc323/Chord-Annotations
    """
    uspop_dir = output_dir / "uspop"
    annotations_dir = uspop_dir / "annotations" / "uspopLabels"

    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }

    # Check if already exists
    if annotations_dir.exists() and any(annotations_dir.rglob("*.lab")):
        print("Skipping UsPop2002 - already exists")
        results["skipped"].append("UsPop2002")
        return results

    uspop_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        repo_dir = tmpdir / "Chord-Annotations"

        if git_clone("https://github.com/tmc323/Chord-Annotations.git", repo_dir):
            try:
                # Copy uspop2002 chord labels
                source_labels = repo_dir / "uspop2002" / "Chord Labels"
                if source_labels.exists():
                    # Create destination
                    annotations_dir.mkdir(parents=True, exist_ok=True)

                    # Copy all .lab files maintaining structure
                    for lab_file in source_labels.rglob("*.lab"):
                        rel_path = lab_file.relative_to(source_labels)
                        dest_path = annotations_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(lab_file, dest_path)

                    # Copy index file if exists
                    index_file = repo_dir / "uspop2002" / "uspopLabels.txt"
                    if index_file.exists():
                        shutil.copy2(index_file, uspop_dir / "annotations" / "uspopLabels.txt")
                    else:
                        # Create index from downloaded files
                        create_uspop_index(annotations_dir, uspop_dir / "annotations" / "uspopLabels.txt")

                    results["success"].append("UsPop2002")
                    print(f"  Installed UsPop2002 annotations")
                else:
                    print(f"  Error: Expected directory not found: {source_labels}")
                    results["failed"].append("UsPop2002")
            except Exception as e:
                print(f"  Error processing UsPop2002: {e}")
                results["failed"].append("UsPop2002")
        else:
            results["failed"].append("UsPop2002")

    return results


def create_uspop_index(labels_dir: Path, index_path: Path):
    """Create index file listing all .lab files."""
    lab_files = sorted(labels_dir.rglob("*.lab"))
    with open(index_path, 'w') as f:
        for lab_file in lab_files:
            rel_path = lab_file.relative_to(labels_dir.parent.parent)
            f.write(f"./{rel_path}\n")
    print(f"  Created index file with {len(lab_files)} entries")


def download_robbiewilliams(output_dir: Path) -> dict:
    """
    Download Robbie Williams annotations.

    Source: http://ispg.deib.polimi.it/mir-software.html
    Note: Direct download URL may change; update as needed.
    """
    rw_dir = output_dir / "robbiewilliams"
    chords_dir = rw_dir / "chords"

    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }

    # Check if already exists
    if chords_dir.exists() and any(chords_dir.rglob("*.txt")):
        print("Skipping Robbie Williams - already exists")
        results["skipped"].append("RobbieWilliams")
        return results

    rw_dir.mkdir(parents=True, exist_ok=True)

    # Try known download URL (may need updating)
    # The POLIMI website provides these as a zip file
    download_url = "http://ispg.deib.polimi.it/software/RWC-Pop-MIDI.zip"

    print("""
Note: Robbie Williams annotations download may fail due to website changes.
If automatic download fails, please:
1. Visit http://ispg.deib.polimi.it/mir-software.html
2. Download the "Robbie Williams" chord annotations
3. Extract to: {}/robbiewilliams/chords/
""".format(output_dir))

    # For now, create placeholder directory structure
    chords_dir.mkdir(parents=True, exist_ok=True)
    (rw_dir / "audio").mkdir(exist_ok=True)

    # Try alternative: check if there's a mirror or GitHub version
    alt_urls = [
        # Add any known mirror URLs here
    ]

    for url in alt_urls:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "rw_annotations.zip"
            if download_file(url, zip_path, "Robbie Williams annotations"):
                try:
                    extract_zip(zip_path, tmpdir / "extracted")
                    # Move to final location
                    for txt_file in (tmpdir / "extracted").rglob("*.txt"):
                        if "README" not in txt_file.name:
                            dest = chords_dir / txt_file.name
                            shutil.copy2(txt_file, dest)
                    results["success"].append("RobbieWilliams")
                    return results
                except Exception as e:
                    print(f"  Error extracting: {e}")

    # Manual download required
    print("""
Robbie Williams annotations must be downloaded manually:
1. Visit: http://ispg.deib.polimi.it/mir-software.html
2. Find "Robbie Williams" dataset
3. Download chord annotations
4. Extract .txt files to: {}/chords/

Expected structure:
  robbiewilliams/
    chords/
      Album1/
        01_SongName_GTChords.txt
        ...
    audio/
      Album1/
        01 - Song Name.mp3
        ...
""".format(rw_dir))

    results["failed"].append("RobbieWilliams (manual download required)")
    return results


def create_directory_structure(output_dir: Path):
    """Create expected directory structure with READMEs."""
    structure = {
        "isophonic": "Place Isophonics annotations and audio here. Structure: Artist/Album/song.{lab,mp3}",
        "uspop": "Place UsPop2002 audio in audio/ subfolder",
        "robbiewilliams": "Place Robbie Williams audio in audio/ subfolder",
    }

    for subdir, readme_content in structure.items():
        dir_path = output_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)

        # Create audio subdirectory
        if subdir in ["uspop", "robbiewilliams"]:
            (dir_path / "audio").mkdir(exist_ok=True)


def print_summary(results: dict, dataset_name: str):
    """Print download summary for a dataset."""
    print(f"\n{dataset_name}:")
    if results["success"]:
        print(f"  Downloaded: {', '.join(results['success'])}")
    if results["skipped"]:
        print(f"  Skipped (already exists): {', '.join(results['skipped'])}")
    if results["failed"]:
        print(f"  Failed: {', '.join(results['failed'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Download chord annotations for BTC training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_annotations.py --output_dir /data/music/chord_recognition
    python download_annotations.py --output_dir ~/datasets/btc_training --dataset isophonics

Note: Audio files must be obtained separately due to copyright restrictions.
After downloading annotations, place audio files in the corresponding directories.
        """
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/datasets/chord_recognition"),
        help="Output directory for annotations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "isophonics", "uspop", "robbiewilliams"],
        help="Which dataset(s) to download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir}")

    # Create directory structure
    create_directory_structure(output_dir)

    all_results = {}

    # Download selected datasets
    if args.dataset in ["all", "isophonics"]:
        print("\n" + "="*50)
        print("Downloading Isophonics annotations...")
        print("="*50)
        all_results["Isophonics"] = download_isophonics(output_dir)

    if args.dataset in ["all", "uspop"]:
        print("\n" + "="*50)
        print("Downloading UsPop2002 annotations...")
        print("="*50)
        all_results["UsPop2002"] = download_uspop(output_dir)

    if args.dataset in ["all", "robbiewilliams"]:
        print("\n" + "="*50)
        print("Downloading Robbie Williams annotations...")
        print("="*50)
        all_results["RobbieWilliams"] = download_robbiewilliams(output_dir)

    # Print summary
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    for dataset_name, results in all_results.items():
        print_summary(results, dataset_name)

    # Print next steps
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("""
1. Obtain audio files (copyright protected, must be acquired separately):

   Isophonics (Beatles, Queen, etc.):
   - Source: CD rips, iTunes, Amazon Music
   - Place in: {}/isophonic/Artist/Album/song.mp3
   - Filenames must match .lab annotation files

   UsPop2002:
   - 195 pop songs from early 2000s
   - Place in: {}/uspop/audio/Artist - Title.mp3
   - Check uspopLabels.txt for expected song list

   Robbie Williams:
   - First 5 studio albums (65 songs)
   - Place in: {}/robbiewilliams/audio/Album/song.mp3

2. Verify dataset completeness:
   python verify_dataset.py --data_dir {}

3. Run source separation:
   python separate_audio.py --data_dir {}

4. Create finetuning dataset:
   python create_finetuning_dataset_labeled.py --data_dir {}
""".format(output_dir, output_dir, output_dir, output_dir, output_dir, output_dir))


if __name__ == "__main__":
    main()
