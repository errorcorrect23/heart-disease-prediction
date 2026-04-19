"""
Heart Disease Dataset Generator
Mimics Cleveland Heart Disease Database (UCI)
303 records, 14 columns (13 features + 1 target)
"""

import pandas as pd
import numpy as np

np.random.seed(42)
n = 303

age      = np.random.randint(29, 78, n)
sex      = np.random.choice([0, 1], n, p=[0.32, 0.68])
cp       = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])
trestbps = np.random.randint(94, 201, n)
chol     = np.random.randint(126, 565, n)
fbs      = np.random.choice([0, 1], n, p=[0.85, 0.15])
restecg  = np.random.choice([0, 1, 2], n, p=[0.50, 0.48, 0.02])
thalach  = np.random.randint(71, 203, n)
exang    = np.random.choice([0, 1], n, p=[0.68, 0.32])
oldpeak  = np.round(np.random.uniform(0.0, 6.2, n), 1)
slope    = np.random.choice([0, 1, 2], n, p=[0.21, 0.46, 0.33])
ca       = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])
thal     = np.random.choice([1, 2, 3], n, p=[0.18, 0.07, 0.75])

# Create realistic target based on risk factors
risk = (
    (age > 55).astype(int) * 0.3 +
    sex * 0.2 +
    (cp == 3).astype(int) * 0.4 +
    (trestbps > 140).astype(int) * 0.2 +
    (chol > 240).astype(int) * 0.15 +
    exang * 0.35 +
    (oldpeak > 2).astype(int) * 0.3 +
    (ca > 0).astype(int) * 0.3 +
    (thal == 3).astype(int) * 0.25
)
prob = 1 / (1 + np.exp(-(risk - 1.2)))
target = (prob > 0.5).astype(int)

df = pd.DataFrame({
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
    'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
    'ca': ca, 'thal': thal, 'target': target
})

df.to_csv('/home/claude/heart_disease_project/data/heart.csv', index=False)
print(f"Dataset saved: {df.shape}")
print(df['target'].value_counts())
