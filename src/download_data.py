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
"""

import os
import sys
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
    
    # Titanic competition dataset identifier
    dataset = "c/titanic"
    
    print(f"Downloading Titanic dataset from Kaggle ({dataset})...")
    print(f"Output directory: {output_path.absolute()}")
    
    try:
        # Download dataset files
        api.competition_download_files(
            competition=dataset,
            path=str(output_path),
            unzip=True
        )
        
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
    
    # Check if files already exist
    output_dir = Path("data/raw")
    train_file = output_dir / "train.csv"
    test_file = output_dir / "test.csv"
    
    if train_file.exists() and test_file.exists():
        print("⚠ Warning: Dataset files already exist!")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        response = input("\nDo you want to re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return
    
    download_titanic_dataset(output_dir=str(output_dir))
    
    # Rename files to match expected names in preprocessing script
    print("\nRenaming files to match expected names...")
    if (output_dir / "train.csv").exists():
        (output_dir / "train.csv").rename(output_dir / "titanic_train.csv")
        print("  ✓ train.csv -> titanic_train.csv")
    
    if (output_dir / "test.csv").exists():
        (output_dir / "test.csv").rename(output_dir / "titanic_test.csv")
        print("  ✓ test.csv -> titanic_test.csv")
    
    print("\n" + "=" * 60)
    print("All done! You can now run your preprocessing script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
