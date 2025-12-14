#!/usr/bin/env python3
"""
Download sample videos from Egocentric-10K dataset for OSHA Vision demo.

Usage:
    python scripts/download_sample.py           # Download 10 clips (default)
    python scripts/download_sample.py --count 5 # Download 5 clips
    python scripts/download_sample.py --all     # Download all available shards

Set HF_TOKEN environment variable or add to .env file for authentication.
"""

import os
import sys
import argparse
import tarfile
from pathlib import Path

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() not in os.environ:
                    os.environ[key.strip()] = value.strip()

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "builddotai/Egocentric-10K"
OUTPUT_DIR = Path("data")

# Known shard paths in the dataset (factory/worker combinations)
SHARD_PATHS = [
    "factory_001/workers/worker_001/factory001_worker001_part00.tar",
    "factory_001/workers/worker_002/factory001_worker002_part00.tar",
    "factory_001/workers/worker_003/factory001_worker003_part00.tar",
    "factory_002/workers/worker_001/factory002_worker001_part00.tar",
    "factory_002/workers/worker_002/factory002_worker002_part00.tar",
    "factory_002/workers/worker_003/factory002_worker003_part00.tar",
    "factory_003/workers/worker_001/factory003_worker001_part00.tar",
    "factory_003/workers/worker_002/factory003_worker002_part00.tar",
    "factory_003/workers/worker_003/factory003_worker003_part00.tar",
    "factory_004/workers/worker_001/factory004_worker001_part00.tar",
    "factory_004/workers/worker_002/factory004_worker002_part00.tar",
    "factory_004/workers/worker_003/factory004_worker003_part00.tar",
]


def extract_mp4s_from_tar(tar_path: str, output_dir: Path, max_clips: int = 1) -> list:
    """Extract MP4 files from a tar archive."""
    extracted = []

    try:
        with tarfile.open(tar_path, "r") as tar:
            mp4_members = [m for m in tar.getmembers() if m.name.endswith(".mp4")]

            for i, member in enumerate(mp4_members[:max_clips]):
                # Extract to temp location
                tar.extract(member, path=output_dir)

                # Get the extracted path and create a cleaner filename
                extracted_path = output_dir / member.name

                # Create clean filename from the path
                # e.g., "factory001_worker001_clip01.mp4"
                clean_name = extracted_path.name
                final_path = output_dir / clean_name

                # Handle nested extraction
                if extracted_path != final_path:
                    if final_path.exists():
                        final_path.unlink()
                    extracted_path.rename(final_path)

                    # Clean up empty parent directories
                    try:
                        extracted_path.parent.rmdir()
                    except OSError:
                        pass

                extracted.append(final_path)
                print(f"  ✓ Extracted: {final_path.name}")

    except Exception as e:
        print(f"  ✗ Error extracting from {tar_path}: {e}")

    return extracted


def download_egocentric_clips(count: int = 10, clips_per_shard: int = 1):
    """Download multiple clips from Egocentric-10K dataset."""

    print("=" * 60)
    print("Egocentric-10K Dataset Downloader")
    print("=" * 60)
    print(f"\nTarget: {count} video clips")

    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and hf_token != "your_token_here":
        print("✓ HF_TOKEN found in environment")
    else:
        print("⚠ No HF_TOKEN found. Set it in .env file or environment:")
        print("  HF_TOKEN=hf_xxx...")
        print()
        hf_token = None

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_extracted = []
    shards_needed = min(len(SHARD_PATHS), (count + clips_per_shard - 1) // clips_per_shard)

    for i, shard_path in enumerate(SHARD_PATHS[:shards_needed]):
        if len(all_extracted) >= count:
            break

        print(f"\n[{i+1}/{shards_needed}] Downloading: {shard_path}")

        try:
            # Download the tar shard with token
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=shard_path,
                repo_type="dataset",
                token=hf_token
            )
            print(f"  Downloaded to cache")

            # Extract MP4s
            remaining = count - len(all_extracted)
            clips_to_extract = min(clips_per_shard, remaining)
            extracted = extract_mp4s_from_tar(local_path, OUTPUT_DIR, clips_to_extract)
            all_extracted.extend(extracted)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    print("\n" + "=" * 60)
    print(f"Download complete! {len(all_extracted)} clips saved to {OUTPUT_DIR}/")
    print("=" * 60)

    if all_extracted:
        print("\nFiles:")
        for f in sorted(all_extracted):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")

    return all_extracted


def list_existing_videos():
    """List MP4 files already in the data directory."""
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*.mp4"))


def main():
    parser = argparse.ArgumentParser(
        description="Download Egocentric-10K video clips for OSHA Vision demo"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of clips to download (default: 10)"
    )
    parser.add_argument(
        "--clips-per-shard", "-c",
        type=int,
        default=1,
        help="Max clips to extract per shard (default: 1)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List existing videos in data/ folder"
    )

    args = parser.parse_args()

    if args.list:
        existing = list_existing_videos()
        if existing:
            print(f"Found {len(existing)} videos in {OUTPUT_DIR}/:")
            for f in existing:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  • {f.name} ({size_mb:.1f} MB)")
        else:
            print(f"No videos found in {OUTPUT_DIR}/")
        return

    download_egocentric_clips(count=args.count, clips_per_shard=args.clips_per_shard)


if __name__ == "__main__":
    main()
