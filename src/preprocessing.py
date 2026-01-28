import pandas as pd
import numpy as np
import os

def load_data_safely(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-16')

def preprocess_titanic():
    raw_train_path = "data/raw/titanic_train.csv"
    raw_test_path = "data/raw/titanic_test.csv"
    output_path = "data/derived/titanic_merged_preprocessed.csv"

    print("--- Loading Titanic Dataset ---")
    if not os.path.exists(raw_train_path) or not os.path.exists(raw_test_path):
        print("Error: Files not found.")
        return

    train = load_data_safely(raw_train_path)
    test = load_data_safely(raw_test_path)
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)

    print("--- Cleaning & Feature Engineering ---")

    # Fill Missing Values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Feature Engineering (Only run if 'Name' exists)
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping).fillna(0)
    
    # Process Sex
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).fillna(0).astype(int)

    # Final cleanup
    cols_to_drop = [c for c in ['Ticket', 'Cabin', 'Name', 'PassengerId'] if c in df.columns]
    df.drop(cols_to_drop, axis=1, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Success! Saved to {output_path}")

if __name__ == "__main__":
    preprocess_titanic()