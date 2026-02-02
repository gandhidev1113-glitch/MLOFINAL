"""
Training script for Titanic survival prediction model.

This script loads preprocessed data, builds a baseline model, evaluates it
with a validation split, and saves the trained model artifact.

Notes:
- Kaggle test set has no labels (Survived), so we evaluate using a validation split
  from the original training portion.
- We one-hot encode categorical columns to avoid sklearn fit errors.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils import load_data_safely


PREPROCESSED_PATH = "data/derived/titanic_merged_preprocessed.csv"
RAW_TRAIN_PATH = "data/raw/titanic_train.csv"


def load_preprocessed_data(preprocessed_path: str = PREPROCESSED_PATH) -> pd.DataFrame | None:
    """Load the preprocessed merged dataset."""
    if not os.path.exists(preprocessed_path):
        print(f"Error: Preprocessed data not found at {preprocessed_path}")
        print("Please run preprocessing first.")
        return None

    df = load_data_safely(preprocessed_path)
    print(f"Loaded preprocessed data: {df.shape}")
    return df


def get_original_train_size(raw_train_path: str = RAW_TRAIN_PATH, default_size: int = 891) -> int:
    """Get the original Kaggle train size to split merged data back into train/test parts."""
    if os.path.exists(raw_train_path):
        train_original = load_data_safely(raw_train_path)
        return len(train_original)
    return default_size


def split_merged_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split merged preprocessed dataframe into original train part and Kaggle test part
    based on raw train size.
    """
    train_size = get_original_train_size()
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    return train_df, test_df


def build_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build feature matrices for training and Kaggle test.

    - Train labels come from 'Survived'
    - Categorical features are one-hot encoded
    - Kaggle test columns are aligned to training columns
    """
    if "Survived" not in train_df.columns:
        raise ValueError("Column 'Survived' is missing in the training portion of the merged data.")

    # Separate X/y for training portion
    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"].astype(int)

    # Kaggle test portion may or may not contain 'Survived' (shouldn't), drop if exists
    X_kaggle = test_df.drop(columns=["Survived"], errors="ignore")

    # One-hot encode to ensure numeric features only
    X = pd.get_dummies(X, drop_first=True)
    X_kaggle = pd.get_dummies(X_kaggle, drop_first=True)

    # Align Kaggle test columns to training columns
    X_kaggle = X_kaggle.reindex(columns=X.columns, fill_value=0)

    # Safety: ensure no NaNs
    X = X.fillna(0)
    X_kaggle = X_kaggle.fillna(0)

    return X, y, X_kaggle


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train a baseline Random Forest classifier."""
    print("\n--- Training Baseline Model ---")
    print("Model: Random Forest Classifier")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    print("Model training completed!")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Evaluate the model and print key metrics."""
    print("\n--- Model Evaluation ---")

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print("\n--- Classification Report (Validation) ---")
    print(classification_report(y_val, y_val_pred))

    print("\n--- Confusion Matrix (Validation) ---")
    print(confusion_matrix(y_val, y_val_pred))

    # Feature importance (top 10)
    print("\n--- Top 10 Feature Importances ---")
    importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(importance.head(10).to_string(index=False))


def save_model(model: RandomForestClassifier, output_path: str = "models/baseline_model.pkl") -> None:
    """Save the trained model artifact."""
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to {output_path}")


def save_kaggle_predictions(
    model: RandomForestClassifier,
    X_kaggle: pd.DataFrame,
    raw_test_path: str = "data/raw/titanic_test.csv",
    output_path: str = "data/output/kaggle_predictions.csv",
) -> None:
    """
    Save Kaggle-style predictions file:
    PassengerId,Survived

    We read PassengerId from the raw Kaggle test file if available.
    """
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    preds = model.predict(X_kaggle).astype(int)

    if os.path.exists(raw_test_path):
        raw_test = load_data_safely(raw_test_path)
        if "PassengerId" in raw_test.columns and len(raw_test) == len(preds):
            sub = pd.DataFrame({"PassengerId": raw_test["PassengerId"].astype(int), "Survived": preds})
        else:
            # Fallback: generate an index-based id if PassengerId is missing/mismatched
            sub = pd.DataFrame({"PassengerId": np.arange(1, len(preds) + 1), "Survived": preds})
    else:
        sub = pd.DataFrame({"PassengerId": np.arange(1, len(preds) + 1), "Survived": preds})

    sub.to_csv(output_path, index=False)
    print(f"✓ Kaggle predictions saved to {output_path}")


def main() -> None:
    """Main training pipeline."""
    print("=" * 60)
    print("Titanic Survival Prediction - Baseline Model Training")
    print("=" * 60)

    df = load_preprocessed_data()
    if df is None:
        return

    # Split merged df back into train/test portions
    train_df, test_df = split_merged_df(df)

    # Build encoded features
    X, y, X_kaggle = build_features(train_df, test_df)

    # Validation split (evaluate properly without Kaggle test labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Kaggle test: {X_kaggle.shape}")

    # Train
    model = train_baseline_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_train, y_train, X_val, y_val)

    # Save model
    save_model(model)

    # Optional: generate Kaggle predictions file
    save_kaggle_predictions(model, X_kaggle)

    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()