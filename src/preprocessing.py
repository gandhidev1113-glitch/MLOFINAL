"""
Data preprocessing pipeline for the Titanic dataset.

This module handles data cleaning, feature engineering, and preparation
for model training.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional

# Handle imports - works from both project root and src directory
try:
    from src.utils import load_data_safely, ensure_dir_exists
except ImportError:
    from utils import load_data_safely, ensure_dir_exists

def preprocess_titanic(
    raw_train_path: str = "data/raw/titanic_train.csv",
    raw_test_path: str = "data/raw/titanic_test.csv",
    output_path: str = "data/derived/titanic_merged_preprocessed.csv"
) -> Optional[pd.DataFrame]:
    """
    Preprocess the Titanic dataset.
    
    This function:
    1. Loads raw train and test datasets
    2. Merges them for consistent preprocessing
    3. Handles missing values
    4. Performs feature engineering
    5. Saves the preprocessed data
    
    Args:
        raw_train_path: Path to raw training data
        raw_test_path: Path to raw test data
        output_path: Path to save preprocessed data
        
    Returns:
        Preprocessed DataFrame, or None if error occurs
    """
    print("--- Loading Titanic Dataset ---")
    
    # Check if input files exist
    if not os.path.exists(raw_train_path) or not os.path.exists(raw_test_path):
        print(f"Error: Files not found.")
        print(f"  Train: {raw_train_path}")
        print(f"  Test: {raw_test_path}")
        return None

    # Load data
    train = load_data_safely(raw_train_path)
    test = load_data_safely(raw_test_path)
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
    
    print(f"Loaded {len(train)} training samples and {len(test)} test samples")

    print("--- Cleaning & Feature Engineering ---")

    # Fill Missing Values
    if 'Age' in df.columns:
        missing_age = df['Age'].isna().sum()
        if missing_age > 0:
            df['Age'] = df['Age'].fillna(df['Age'].median())
            print(f"  Filled {missing_age} missing Age values")
    
    if 'Fare' in df.columns:
        missing_fare = df['Fare'].isna().sum()
        if missing_fare > 0:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            print(f"  Filled {missing_fare} missing Fare values")
    
    if 'Embarked' in df.columns:
        missing_embarked = df['Embarked'].isna().sum()
        if missing_embarked > 0:
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
            print(f"  Filled {missing_embarked} missing Embarked values")

    # Feature Engineering: Extract Title from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace([
            'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'
        ], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping).fillna(0)
        print("  Extracted and encoded Title feature")
    
    # Process Sex: Encode as binary
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).fillna(0).astype(int)
        print("  Encoded Sex feature")

    # Final cleanup: Drop unnecessary columns
    cols_to_drop = [c for c in ['Ticket', 'Cabin', 'Name', 'PassengerId'] if c in df.columns]
    if cols_to_drop:
        df.drop(cols_to_drop, axis=1, inplace=True)
        print(f"  Dropped columns: {', '.join(cols_to_drop)}")

    # Save preprocessed data
    ensure_dir_exists(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    print(f"\n✓ Success! Saved preprocessed data to {output_path}")
    print(f"  Final shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    preprocess_titanic()