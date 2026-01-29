"""
Training script for Titanic survival prediction model.

This script loads preprocessed data, trains a baseline model, and saves it.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils import load_data_safely


def load_preprocessed_data():
    """Load the preprocessed dataset."""
    preprocessed_path = "data/derived/titanic_merged_preprocessed.csv"
    
    if not os.path.exists(preprocessed_path):
        print(f"Error: Preprocessed data not found at {preprocessed_path}")
        print("Please run preprocessing first: python src/preprocessing.py")
        return None
    
    df = load_data_safely(preprocessed_path)
    print(f"Loaded preprocessed data: {df.shape}")
    return df


def split_train_test(df):
    """
    Split the merged dataset back into train and test sets.
    The original train set has 891 rows (with 'Survived' column).
    """
    # Load original train to get the split point
    raw_train_path = "data/raw/titanic_train.csv"
    if os.path.exists(raw_train_path):
        train_original = load_data_safely(raw_train_path)
        train_size = len(train_original)
    else:
        # Default split: 891 rows for training (standard Titanic dataset)
        train_size = 891
    
    # Check if 'Survived' column exists
    if 'Survived' not in df.columns:
        print("Error: 'Survived' column not found in preprocessed data.")
        return None, None, None, None
    
    # Split based on original train size
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # Separate features and target
    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop('Survived', axis=1) if 'Survived' in test_df.columns else None
    y_test = test_df['Survived'] if 'Survived' in test_df.columns else None
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape if X_test is not None else 'N/A'}")
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train, y_train):
    """Train a baseline Random Forest classifier."""
    print("\n--- Training Baseline Model ---")
    print("Model: Random Forest Classifier")
    
    # Initialize model with reasonable defaults
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    return model


def evaluate_model(model, X_train, y_train, X_test=None, y_test=None):
    """Evaluate the model and print metrics."""
    print("\n--- Model Evaluation ---")
    
    # Training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Test accuracy (if available)
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_test_pred))
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, y_test_pred))
    
    # Feature importance
    print("\n--- Top 5 Feature Importances ---")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head())


def save_model(model, output_path="models/baseline_model.pkl"):
    """Save the trained model."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved to {output_path}")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Titanic Survival Prediction - Baseline Model Training")
    print("=" * 60)
    
    # Load data
    df = load_preprocessed_data()
    if df is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_test(df)
    if X_train is None:
        return
    
    # Train model
    model = train_baseline_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Save model
    save_model(model)
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

