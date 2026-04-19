"""
Heart Disease Prediction Project
Data Structure Module
Based on Cleveland Heart Disease Database (UCI)
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  DATA SCHEMA DEFINITION
# ─────────────────────────────────────────────

FEATURE_SCHEMA = {
    "age": {
        "type": "int",
        "description": "Age in years",
        "range": (29, 77),
        "category": "demographic"
    },
    "sex": {
        "type": "binary",
        "description": "Sex (1 = male, 0 = female)",
        "values": [0, 1],
        "category": "demographic"
    },
    "cp": {
        "type": "categorical",
        "description": "Chest pain type",
        "values": {0: "Typical angina", 1: "Atypical angina",
                   2: "Non-anginal pain", 3: "Asymptomatic"},
        "category": "clinical"
    },
    "trestbps": {
        "type": "int",
        "description": "Resting blood pressure (mm Hg on admission)",
        "range": (94, 200),
        "unit": "mm Hg",
        "category": "clinical"
    },
    "chol": {
        "type": "int",
        "description": "Serum cholesterol in mg/dl",
        "range": (126, 564),
        "unit": "mg/dl",
        "category": "clinical"
    },
    "fbs": {
        "type": "binary",
        "description": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "values": [0, 1],
        "category": "clinical"
    },
    "restecg": {
        "type": "categorical",
        "description": "Resting electrocardiographic results",
        "values": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
        "category": "clinical"
    },
    "thalach": {
        "type": "int",
        "description": "Maximum heart rate achieved",
        "range": (71, 202),
        "unit": "bpm",
        "category": "clinical"
    },
    "exang": {
        "type": "binary",
        "description": "Exercise induced angina (1 = yes, 0 = no)",
        "values": [0, 1],
        "category": "clinical"
    },
    "oldpeak": {
        "type": "float",
        "description": "ST depression induced by exercise relative to rest",
        "range": (0.0, 6.2),
        "category": "clinical"
    },
    "slope": {
        "type": "categorical",
        "description": "Slope of the peak exercise ST segment",
        "values": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        "category": "clinical"
    },
    "ca": {
        "type": "int",
        "description": "Number of major vessels (0-3) colored by fluoroscopy",
        "range": (0, 3),
        "category": "imaging"
    },
    "thal": {
        "type": "categorical",
        "description": "Thalassemia",
        "values": {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"},
        "category": "imaging"
    },
    "target": {
        "type": "binary",
        "description": "Heart disease diagnosis (1 = disease, 0 = no disease)",
        "values": [0, 1],
        "category": "label"
    }
}

# ─────────────────────────────────────────────
#  SAMPLE DATA GENERATION (mimics Cleveland DB)
# ─────────────────────────────────────────────

def generate_sample_data(n_samples=303, random_state=42):
    """Generate a synthetic dataset matching Cleveland Heart Disease structure."""
    np.random.seed(random_state)
    n = n_samples

    data = {
        "age":      np.random.randint(29, 78, n),
        "sex":      np.random.choice([0, 1], n, p=[0.32, 0.68]),
        "cp":       np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08]),
        "trestbps": np.random.randint(94, 201, n),
        "chol":     np.random.randint(126, 565, n),
        "fbs":      np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "restecg":  np.random.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02]),
        "thalach":  np.random.randint(71, 203, n),
        "exang":    np.random.choice([0, 1], n, p=[0.68, 0.32]),
        "oldpeak":  np.round(np.random.uniform(0.0, 6.2, n), 1),
        "slope":    np.random.choice([0, 1, 2], n, p=[0.21, 0.46, 0.33]),
        "ca":       np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07]),
        "thal":     np.random.choice([1, 2, 3], n, p=[0.18, 0.07, 0.75]),
        "target":   np.random.choice([0, 1], n, p=[0.4554, 0.5446]),
    }

    return pd.DataFrame(data)


# ─────────────────────────────────────────────
#  DATA VALIDATION
# ─────────────────────────────────────────────

def validate_dataframe(df):
    """Validate dataset against schema."""
    issues = []

    required_cols = list(FEATURE_SCHEMA.keys())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        issues.append(f"Null values found in: {null_cols}")

    for col in ["sex", "fbs", "exang", "target"]:
        if col in df.columns:
            invalid = df[~df[col].isin([0, 1])][col]
            if not invalid.empty:
                issues.append(f"Invalid binary values in '{col}'")

    return issues if issues else ["✓ Dataset is valid"]


# ─────────────────────────────────────────────
#  DATA SUMMARY
# ─────────────────────────────────────────────

def print_data_summary(df):
    print("=" * 55)
    print("  HEART DISEASE DATASET — STRUCTURAL SUMMARY")
    print("=" * 55)
    print(f"  Total Records  : {len(df)}")
    print(f"  Total Features : {len(df.columns) - 1} (+ 1 target)")
    print(f"  Positive Cases : {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"  Negative Cases : {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
    print(f"  Missing Values : {df.isnull().sum().sum()}")
    print("-" * 55)
    print("\n  FEATURE BREAKDOWN BY CATEGORY:")
    from collections import defaultdict
    cats = defaultdict(list)
    for feat, meta in FEATURE_SCHEMA.items():
        if feat != "target":
            cats[meta["category"]].append(feat)
    for cat, feats in cats.items():
        print(f"  [{cat.upper()}] {', '.join(feats)}")
    print("\n  SCHEMA DETAILS:")
    print(f"  {'Feature':<12} {'Type':<12} {'Description'}")
    print(f"  {'-'*12} {'-'*12} {'-'*30}")
    for feat, meta in FEATURE_SCHEMA.items():
        print(f"  {feat:<12} {meta['type']:<12} {meta['description'][:40]}")
    print("=" * 55)

    validation = validate_dataframe(df)
    print("\n  VALIDATION RESULTS:")
    for v in validation:
        print(f"  {v}")
    print("=" * 55)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_sample_data(n_samples=303)
    print_data_summary(df)

    # Save to CSV
    df.to_csv("/home/claude/heart_disease_project/heart_dataset.csv", index=False)
    print("\n  ✓ Dataset saved to heart_dataset.csv")

    # Show first few rows
    print("\n  SAMPLE RECORDS (first 5 rows):")
    print(df.head().to_string())
