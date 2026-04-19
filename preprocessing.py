"""
Module 1: Data Pre-Processing
Heart Disease Prediction Project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data(path='data/heart.csv'):
    """Load CSV dataset."""
    df = pd.read_csv(path)
    print(f"[✓] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def quality_check(df):
    """Run data quality checks."""
    print("\n── DATA QUALITY REPORT ──────────────────────")
    print(f"  Shape            : {df.shape}")
    print(f"  Null values      : {df.isnull().sum().sum()}")
    print(f"  Duplicate rows   : {df.duplicated().sum()}")
    print(f"  Data types       : {df.dtypes.value_counts().to_dict()}")
    print(f"\n  Target distribution:")
    vc = df['target'].value_counts()
    for k, v in vc.items():
        label = "Heart Disease" if k == 1 else "No Disease   "
        print(f"    {label} (={k}): {v} ({v/len(df)*100:.1f}%)")
    print("─────────────────────────────────────────────")
    return df.isnull().sum().sum() == 0


def handle_missing(df):
    """Fill any missing values with column median."""
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[✓] Removed {before - len(df)} duplicate rows")
    return df


def preprocess(df):
    """Full preprocessing pipeline."""
    df = handle_missing(df)
    df = remove_duplicates(df)

    X = df.drop('target', axis=1)
    y = df['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=0, stratify=y
    )

    print(f"[✓] Train set : {X_train.shape[0]} samples")
    print(f"[✓] Test set  : {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, scaler


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = load_data()
    quality_check(df)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print("\n[✓] Preprocessing complete.")
