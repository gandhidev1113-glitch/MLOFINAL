"""
Utility functions for data loading and common operations.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_data_safely(path: str) -> pd.DataFrame:
    """
    Load CSV file with automatic encoding detection.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-16')


def ensure_dir_exists(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to project root
    """
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

