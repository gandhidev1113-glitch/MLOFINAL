r"""
Download script for fetching the Titanic dataset from Kaggle using the Kaggle API.

Prerequisites:
1. Install Kaggle API: pip install kaggle (or via project dependencies)
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Scroll to "API" section and click "Create New Token"
   - This downloads a kaggle.json file
   - Place it in one of these locations:
     * Windows: C:\Users\<username>\.kaggle\kaggle.json
     * Linux/Mac: ~/.kaggle/kaggle.json
   - Set appropriate permissions (chmod 600 on Linux/Mac)

Alternatively, set environment variables:
   - KAGGLE_USERNAME
   - KAGGLE_KEY

Optional behavior:
- By default, if dataset files already exist, the script will skip downloading.
- To force re-download, set environment variable:
    FORCE_DOWNLOAD=1
"""

import os
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def setup_kaggle_api():
    """Initialize and authenticate Kaggle API."""
    api = KaggleApi()
    try:
        api.authenticate()
        return api
    except Exception as e:
        print("Error: Failed to authenticate with Kaggle API.")
        print(f"Details: {e}")
        print("\nPlease ensure you have:")
        print("1. Installed kaggle package: pip install kaggle")
        print("2. Set up credentials (kaggle.json or environment variables)")
        print("   See script header for instructions.")
        sys.exit(1)


def download_titanic_dataset(output_dir="data/raw"):
    """
    Download the Titanic dataset from Kaggle.

    Args:
        output_dir: Directory where the dataset files will be saved.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Connecting to Kaggle API...")
    api = setup_kaggle_api()

    # Titanic competition slug
    dataset = "titanic"

    print(f"Downloading Titanic dataset from Kaggle ({dataset})...")
    print(f"Output directory: {output_path.absolute()}")

    try:
        # Download dataset files (Kaggle API downloads a zip file for competitions)
        api.competition_download_files(
            competition=dataset,
            path=str(output_path),
        )

        # Unzip downloaded archive (some kaggle versions do not support unzip=True)
        zip_path = output_path / f"{dataset}.zip"
        if zip_path.exists():
            print(f"Unzipping: {zip_path.name} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(output_path)
            # Remove zip after extraction to keep folder clean
            zip_path.unlink()
        else:
            print("Warning: zip file not found after download.")
            print("If files are not present, please check Kaggle API output.")

        print("\n✓ Download completed successfully!")
        print(f"Files saved to: {output_path.absolute()}")

        # List downloaded files
        downloaded_files = list(output_path.glob("*.csv"))
        if downloaded_files:
            print("\nDownloaded files:")
            for file in downloaded_files:
                print(f"  - {file.name}")

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Verify you have accepted the competition rules on Kaggle")
        print("2. Check your internet connection")
        print("3. Ensure the dataset identifier is correct")
        sys.exit(1)


def main():
    """Main function to run the download script."""
    print("=" * 60)
    print("Kaggle Titanic Dataset Download Script")
    print("=" * 60)
    print()

    output_dir = Path("data/raw")

    # Accept both original Kaggle names and renamed names
    train_candidates = [output_dir / "train.csv", output_dir / "titanic_train.csv"]
    test_candidates = [output_dir / "test.csv", output_dir / "titanic_test.csv"]

    data_exists = any(p.exists() for p in train_candidates) and any(p.exists() for p in test_candidates)

    force_download = os.getenv("FORCE_DOWNLOAD", "0").strip() == "1"

    if data_exists and not force_download:
        print("⚠ Warning: Dataset files already exist!")
        for p in train_candidates + test_candidates:
            if p.exists():
                print(f"  - {p}")
        print("\nSkipping download. Set FORCE_DOWNLOAD=1 to re-download.")
        return

    download_titanic_dataset(output_dir=str(output_dir))

        # Rename files to match expected names in preprocessing script
    print("\nRenaming files to match expected names...")

    train_src = output_dir / "train.csv"
    test_src = output_dir / "test.csv"
    train_dst = output_dir / "titanic_train.csv"
    test_dst = output_dir / "titanic_test.csv"

    # --- train ---
    if train_src.exists():
        if train_dst.exists():
            if force_download:
                train_dst.unlink()  # remove old file to allow overwrite
                print("  ✓ overwriting titanic_train.csv")
            else:
                print("  ✓ titanic_train.csv already present (no overwrite)")
                # Optionally delete the newly downloaded train.csv to avoid confusion
                # train_src.unlink()
        if not train_dst.exists():
            train_src.rename(train_dst)
            print("  ✓ train.csv -> titanic_train.csv")
    elif train_dst.exists():
        print("  ✓ titanic_train.csv already present")

    # --- test ---
    if test_src.exists():
        if test_dst.exists():
            if force_download:
                test_dst.unlink()
                print("  ✓ overwriting titanic_test.csv")
            else:
                print("  ✓ titanic_test.csv already present (no overwrite)")
                # Optionally delete the newly downloaded test.csv to avoid confusion
                # test_src.unlink()
        if not test_dst.exists():
            test_src.rename(test_dst)
            print("  ✓ test.csv -> titanic_test.csv")
    elif test_dst.exists():
        print("  ✓ titanic_test.csv already present")


    print("\n" + "=" * 60)
    print("All done! You can now run your preprocessing script.")
    print("=" * 60)


if __name__ == "__main__":
    main()