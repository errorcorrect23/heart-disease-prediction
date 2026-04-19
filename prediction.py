"""
Module 4: Prediction
Load saved models and predict on new patient data
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

CP_MAP    = {0: 'Typical Angina', 1: 'Atypical Angina',
             2: 'Non-anginal Pain', 3: 'Asymptomatic'}
THAL_MAP  = {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}
SLOPE_MAP = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}


def load_models(model_dir='models'):
    rf     = joblib.load(f'{model_dir}/random_forest.pkl')
    lr     = joblib.load(f'{model_dir}/logistic_regression.pkl')
    scaler = joblib.load(f'{model_dir}/scaler.pkl')
    return rf, lr, scaler


def predict_single(patient_data: dict, model_dir='models'):
    """
    Predict heart disease for a single patient.

    Parameters
    ----------
    patient_data : dict with keys matching FEATURE_COLUMNS
    model_dir    : path to saved model files

    Returns
    -------
    dict with prediction details
    """
    rf, lr, scaler = load_models(model_dir)

    df = pd.DataFrame([patient_data])[FEATURE_COLUMNS]
    X_scaled = scaler.transform(df)

    rf_pred  = rf.predict(X_scaled)[0]
    rf_prob  = rf.predict_proba(X_scaled)[0]
    lr_pred  = lr.predict(X_scaled)[0]
    lr_prob  = lr.predict_proba(X_scaled)[0]

    result = {
        'random_forest': {
            'prediction': int(rf_pred),
            'label': 'Heart Disease Detected' if rf_pred == 1 else 'No Heart Disease',
            'probability_disease': round(rf_prob[1] * 100, 2),
            'probability_healthy': round(rf_prob[0] * 100, 2),
        },
        'logistic_regression': {
            'prediction': int(lr_pred),
            'label': 'Heart Disease Detected' if lr_pred == 1 else 'No Heart Disease',
            'probability_disease': round(lr_prob[1] * 100, 2),
            'probability_healthy': round(lr_prob[0] * 100, 2),
        },
        'ensemble_vote': int(round((rf_pred + lr_pred) / 2)),
        'risk_level': (
            'HIGH' if rf_prob[1] > 0.7 else
            'MODERATE' if rf_prob[1] > 0.4 else 'LOW'
        )
    }
    return result


def predict_batch(csv_path: str, model_dir='models', out_path='outputs/batch_predictions.csv'):
    """Predict for a CSV file of patients."""
    rf, lr, scaler = load_models(model_dir)
    df = pd.read_csv(csv_path)

    X = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    df['rf_prediction']  = rf.predict(X_scaled)
    df['rf_probability'] = rf.predict_proba(X_scaled)[:, 1].round(4)
    df['lr_prediction']  = lr.predict(X_scaled)
    df['lr_probability'] = lr.predict_proba(X_scaled)[:, 1].round(4)
    df['risk_level'] = df['rf_probability'].apply(
        lambda p: 'HIGH' if p > 0.7 else 'MODERATE' if p > 0.4 else 'LOW'
    )

    df.to_csv(out_path, index=False)
    print(f"[✓] Batch predictions saved → {out_path}")
    print(f"    HIGH risk     : {(df['risk_level']=='HIGH').sum()}")
    print(f"    MODERATE risk : {(df['risk_level']=='MODERATE').sum()}")
    print(f"    LOW risk      : {(df['risk_level']=='LOW').sum()}")
    return df


def print_result(result: dict, patient: dict):
    """Pretty-print a prediction result."""
    print("\n" + "="*50)
    print("  HEART DISEASE PREDICTION REPORT")
    print("="*50)
    print("  PATIENT DATA:")
    print(f"    Age          : {patient.get('age')} years")
    print(f"    Sex          : {'Male' if patient.get('sex') == 1 else 'Female'}")
    print(f"    Chest Pain   : {CP_MAP.get(patient.get('cp'), 'Unknown')}")
    print(f"    Rest BP      : {patient.get('trestbps')} mm Hg")
    print(f"    Cholesterol  : {patient.get('chol')} mg/dl")
    print(f"    Max HR       : {patient.get('thalach')} bpm")
    print(f"    Ex. Angina   : {'Yes' if patient.get('exang') == 1 else 'No'}")
    print(f"    ST Depression: {patient.get('oldpeak')}")
    print(f"    Thalassemia  : {THAL_MAP.get(patient.get('thal'), 'Unknown')}")
    print("-"*50)
    rf = result['random_forest']
    lr = result['logistic_regression']
    print(f"  RANDOM FOREST  : {rf['label']}")
    print(f"    → Disease prob : {rf['probability_disease']}%")
    print(f"    → Healthy prob : {rf['probability_healthy']}%")
    print(f"  LOGISTIC REG.  : {lr['label']}")
    print(f"    → Disease prob : {lr['probability_disease']}%")
    print(f"    → Healthy prob : {lr['probability_healthy']}%")
    print("-"*50)
    risk_colors = {'HIGH': '⚠️ ', 'MODERATE': '⚡ ', 'LOW': '✅ '}
    print(f"  RISK LEVEL     : {risk_colors.get(result['risk_level'],'')} {result['risk_level']}")
    print("="*50)


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Example patient
    sample_patient = {
        'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 145,
        'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
        'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }

    result = predict_single(sample_patient)
    print_result(result, sample_patient)

    # Batch prediction on the full dataset
    predict_batch('data/heart.csv')
