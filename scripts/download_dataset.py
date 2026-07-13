"""
Downloads a landmark dataset from Kaggle using the official Kaggle API.

SETUP (one-time):
1. Create a Kaggle account -> https://www.kaggle.com
2. Go to Account settings -> API -> "Create New Token".
   This downloads a file called `kaggle.json`.
3. Place it at:
     Windows:  C:\\Users\\<you>\\.kaggle\\kaggle.json
     Mac/Linux: ~/.kaggle/kaggle.json
4. Run: pip install -r requirements.txt   (already includes the `kaggle` package)

USAGE:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --slug "some-user/some-other-landmark-dataset"

This will download + unzip the dataset into data/raw/, ready for
scripts/prepare_data.py to split into train/val/test.
"""

import argparse
import os
import sys
import yaml
import zipfile


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Download a landmark dataset from Kaggle")
    parser.add_argument("--slug", type=str, default=None,
                         help="Kaggle dataset slug, e.g. 'kayvanshah/landmarks-dataset'. "
                              "Defaults to the slug in configs/config.yaml")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    slug = args.slug or cfg["dataset"]["kaggle_slug"]
    raw_dir = cfg["dataset"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        print("\nERROR: Kaggle API credentials not found.")
        print("Make sure ~/.kaggle/kaggle.json (or %HOMEPATH%\\.kaggle\\kaggle.json on Windows) exists.")
        print(f"Details: {e}\n")
        sys.exit(1)

    print(f"Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset '{slug}' into '{raw_dir}' ...")
    api.dataset_download_files(slug, path=raw_dir, unzip=True, quiet=False)

    print("\nDone. Contents of data/raw/:")
    for item in sorted(os.listdir(raw_dir))[:20]:
        print(f"  - {item}")

    print(
        "\nNext step: inspect data/raw/ to find the folder containing one "
        "sub-folder per landmark class, then run:\n"
        "    python scripts/prepare_data.py --source data/raw/<inner-folder-if-any>\n"
    )


if __name__ == "__main__":
    main()
