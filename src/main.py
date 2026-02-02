"""
Main entry point for the ML Ops Titanic Survival Prediction project.

This script provides a command-line interface to run different components
of the ML pipeline.
"""

import argparse
import sys
from pathlib import Path


def run_preprocessing():
    """Run the data preprocessing pipeline."""
    from .preprocessing import preprocess_titanic
    print("Running preprocessing pipeline...\n")
    result = preprocess_titanic()
    if result is None:
        sys.exit(1)


def run_training():
    """Run the model training pipeline."""
    from .train import main as train_main
    print("Running training pipeline...\n")
    train_main()


def run_download():
    """Run the data download script."""
    from .download_data import main as download_main
    print("Running data download...\n")
    download_main()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="ML Ops Titanic Survival Prediction - Main Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py preprocess    # Run data preprocessing
  python main.py train         # Train the model
  python main.py download      # Download dataset from Kaggle
  python main.py all           # Run full pipeline (download -> preprocess -> train)
        """
    )
    
    parser.add_argument(
        'command',
        choices=['preprocess', 'train', 'download', 'all'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ML Ops Final Project - Titanic Survival Prediction")
    print("=" * 60)
    print()
    
    if args.command == 'download':
        run_download()
    elif args.command == 'preprocess':
        run_preprocessing()
    elif args.command == 'train':
        run_training()
    elif args.command == 'all':
        print("Running full pipeline: download -> preprocess -> train\n")
        run_download()
        print("\n" + "-" * 60 + "\n")
        run_preprocessing()
        print("\n" + "-" * 60 + "\n")
        run_training()
    
    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
