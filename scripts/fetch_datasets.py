"""Download Loghub datasets for all configured OS types.

Usage:
    python scripts/fetch_datasets.py [--os linux windows mac network]

Downloads 2k-sample files from the Loghub GitHub repo.
For full datasets (BGL, Thunderbird), prints the Zenodo DOI download link.
"""
import argparse
import sys
from pathlib import Path

import requests

BASE_RAW = "https://raw.githubusercontent.com/logpai/loghub/master"

# (loghub_subfolder, filename, local_subfolder)
DATASETS = {
    "linux": [
        ("BGL", "BGL_2k.log", "BGL.log"),
    ],
    "windows": [
        ("Windows", "Windows_2k.log", "Windows.log"),
    ],
    "mac": [
        ("Mac", "Mac_2k.log", "Mac.log"),
    ],
    "network": [
        ("OpenStack", "OpenStack_2k.log", "openstack.log"),
    ],
}

FULL_DATASET_NOTES = {
    "linux": (
        "Full BGL dataset (4.7M lines): "
        "https://zenodo.org/record/3227177  — download BGL.tar.gz"
    ),
    "windows": "Loghub Windows full: https://github.com/logpai/loghub (Windows/ folder)",
    "mac": "Loghub Mac full: https://github.com/logpai/loghub (Mac/ folder)",
    "network": (
        "Full OpenStack or Thunderbird: "
        "https://zenodo.org/record/3227177"
    ),
}


def download(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        print(f"  ✓ {dest}")
        return True
    except Exception as e:
        print(f"  ✗ {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Loghub datasets")
    parser.add_argument(
        "--os",
        nargs="+",
        default=["linux", "windows", "mac", "network"],
        choices=["linux", "windows", "mac", "network"],
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent / "data" / "raw"),
        help="Root directory for raw data",
    )
    args = parser.parse_args()

    root = Path(args.root)

    for os_type in args.os:
        print(f"\n[{os_type.upper()}]")
        entries = DATASETS.get(os_type, [])
        for subfolder, remote_file, local_name in entries:
            url = f"{BASE_RAW}/{subfolder}/{remote_file}"
            dest = root / os_type / local_name
            if dest.exists():
                print(f"  Already exists: {dest}")
                continue
            download(url, dest)

        if os_type in FULL_DATASET_NOTES:
            print(f"  NOTE (full dataset): {FULL_DATASET_NOTES[os_type]}")

    print("\nDone. Place full dataset files in data/raw/<os_type>/ and rerun.")


if __name__ == "__main__":
    main()
