#!/usr/bin/env python
"""Acquire the PTB-XL dataset (100 Hz subset) for diagnostic training.

Usage:
    python scripts/get_ptbxl.py            # extract the 100 Hz records + metadata
    python scripts/get_ptbxl.py --download # also download the zip if missing

Only the 100 Hz records (records100/) and the two metadata CSVs are extracted;
the 500 Hz set is skipped to save ~1.5 GB. The dataset stays git-ignored.
"""
import argparse
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTBXL_DIR = os.path.join(ROOT, "data", "ptbxl")
ZIP_PATH = os.path.join(PTBXL_DIR, "ptbxl-1.0.3.zip")
ZIP_URL = ("https://physionet.org/static/published-projects/ptb-xl/"
           "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip")

KEEP = ("records100/", "ptbxl_database.csv", "scp_statements.csv")


def download():
    os.makedirs(PTBXL_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        print(f"Zip already present: {ZIP_PATH}")
        return
    print(f"Downloading PTB-XL (~1.7 GB) from PhysioNet ...\n{ZIP_URL}")
    import urllib.request
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print("Download complete.")


def extract():
    if not os.path.exists(ZIP_PATH):
        print(f"Zip not found at {ZIP_PATH}.\n"
              f"Download it with:\n"
              f"  curl -L -C - -o \"{ZIP_PATH}\" \"{ZIP_URL}\"\n"
              f"or re-run this script with --download.")
        sys.exit(1)

    print(f"Extracting 100 Hz records + metadata from {ZIP_PATH} ...")
    n = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.namelist():
            if any(k in member for k in KEEP):
                zf.extract(member, PTBXL_DIR)
                n += 1
                if n % 5000 == 0:
                    print(f"  extracted {n} files")
    print(f"Done. Extracted {n} files into {PTBXL_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="Download the zip if missing.")
    args = ap.parse_args()
    if args.download:
        download()
    extract()
