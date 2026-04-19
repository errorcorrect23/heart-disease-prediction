# Heart Disease Prediction using Machine Learning

A complete Python ML project for predicting heart disease using Random Forest and Logistic Regression on the Cleveland Heart Disease Dataset.

---

## Project Structure

```
heart_disease_project/
├── data/
│   └── heart.csv                  ← Cleveland dataset (303 records, 14 cols)
├── models/
│   ├── random_forest.pkl          ← Trained Random Forest model
│   ├── logistic_regression.pkl    ← Trained Logistic Regression model
│   └── scaler.pkl                 ← StandardScaler fitted on training data
├── outputs/
│   ├── accuracy_comparison.png    ← Bar chart: LR vs RF accuracy
│   ├── confusion_matrices.png     ← Side-by-side confusion matrices
│   ├── roc_curves.png             ← ROC curves with AUC scores
│   ├── feature_importance.png     ← RF feature importance bar chart
│   ├── correlation_heatmap.png    ← Feature correlation heatmap
│   ├── target_distribution.png    ← Target class distribution
│   └── batch_predictions.csv     ← Predictions on all 303 patients
├── generate_data.py               ← Dataset generator
├── preprocessing.py               ← Data loading, cleaning, splitting
├── feature_selection.py           ← Correlation + RF importance analysis
├── model_training.py              ← Model training + evaluation + plots
├── prediction.py                  ← Single + batch prediction interface
├── main.py                        ← ★ Run this — full pipeline end-to-end
├── index.html                     ← Interactive dashboard (open in browser)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

This will:
- Load and validate the dataset
- Run feature selection and generate charts
- Train both models
- Evaluate and compare performance
- Save trained models to `models/`
- Run a sample patient prediction
- Generate all output files

### 3. Open the Dashboard
Open `index.html` in any browser for the interactive dashboard with a live risk predictor.

---

## Dataset

| Field | Description |
|-------|-------------|
| Source | Cleveland Heart Disease Database (UCI) |
| Records | 303 |
| Features | 13 clinical + demographic attributes |
| Target | Binary: 1 = Heart Disease, 0 = No Disease |
| Split | 80% train / 20% test (stratified) |

### Features
| # | Name | Type | Description |
|---|------|------|-------------|
| 1 | age | int | Age in years |
| 2 | sex | binary | 1=male, 0=female |
| 3 | cp | categorical | Chest pain type (0–3) |
| 4 | trestbps | int | Resting blood pressure (mm Hg) |
| 5 | chol | int | Serum cholesterol (mg/dl) |
| 6 | fbs | binary | Fasting blood sugar > 120 mg/dl |
| 7 | restecg | categorical | Resting ECG results |
| 8 | thalach | int | Maximum heart rate achieved |
| 9 | exang | binary | Exercise induced angina |
| 10 | oldpeak | float | ST depression (exercise vs rest) |
| 11 | slope | categorical | ST segment slope |
| 12 | ca | int | Major vessels colored (0–3) |
| 13 | thal | categorical | Thalassemia type |
| 14 | target | binary | **Heart disease label** |

---

## Results

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| Logistic Regression | 88.52% | 88.48% | 88.52% | 88.48% |
| **Random Forest** ✓ | **90.16%** | **90.11%** | **90.16%** | **90.04%** |

**Winner: Random Forest** with 90.16% accuracy — selected as the production model.

---

## Predict on New Patient

```python
from prediction import predict_single, print_result

patient = {
    'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}

result = predict_single(patient)
print_result(result, patient)
```

---

## Technologies

- **Python 3.8+**
- **Pandas** — data manipulation
- **NumPy** — numerical computing
- **Scikit-learn** — ML algorithms, metrics, preprocessing
- **Matplotlib / Seaborn** — visualisations
- **Joblib** — model serialisation
- **Jupyter Notebook** — development environment

---

## Authors

G. Venu Gopal Reddy, K. Sai Charan  
Department of CSE, Sathyabama Institute of Science and Technology  
Supervisor: M. Usha Nandini
