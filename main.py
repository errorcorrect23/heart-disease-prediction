"""
main.py — Full Pipeline Runner
Heart Disease Prediction using Machine Learning
Run this to execute the entire project end-to-end.
"""

import os
import sys

# Ensure we run from the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  HEART DISEASE PREDICTION — FULL PIPELINE")
print("  Random Forest + Logistic Regression")
print("=" * 55)

# ── STEP 1: Generate / Load Dataset ────────────────────────────
print("\n[STEP 1] Loading Dataset...")
from preprocessing import load_data, quality_check, preprocess

df = load_data('data/heart.csv')
quality_check(df)
X_train, X_test, y_train, y_test, scaler = preprocess(df)

# ── STEP 2: Feature Selection ───────────────────────────────────
print("\n[STEP 2] Feature Selection...")
from feature_selection import (correlation_analysis, select_top_features,
                                plot_feature_importance, plot_correlation_heatmap)

correlation_analysis(df)
top_features, importance = select_top_features(X_train, y_train, k=10)
plot_feature_importance(importance)
plot_correlation_heatmap(df)

# ── STEP 3: Train Models ────────────────────────────────────────
print("\n[STEP 3] Training Models...")
from model_training import (train_logistic_regression, train_random_forest,
                             evaluate_model, plot_confusion_matrices,
                             plot_accuracy_comparison, plot_roc_curves,
                             plot_target_distribution, save_models)

plot_target_distribution(df['target'])
lr = train_logistic_regression(X_train, y_train)
rf = train_random_forest(X_train, y_train, search_best=True)

# ── STEP 4: Evaluate ────────────────────────────────────────────
print("\n[STEP 4] Evaluating Models...")
acc_lr, cm_lr, cr_lr, _ = evaluate_model(lr, X_test, y_test, "Logistic Regression")
acc_rf, cm_rf, cr_rf, _ = evaluate_model(rf, X_test, y_test, "Random Forest")

plot_confusion_matrices(cm_lr, cm_rf, acc_lr, acc_rf)
plot_accuracy_comparison(acc_lr, acc_rf)
plot_roc_curves(lr, rf, X_test, y_test)

# ── STEP 5: Save Models ─────────────────────────────────────────
print("\n[STEP 5] Saving Models...")
save_models(lr, rf, scaler)

# ── STEP 6: Sample Prediction ───────────────────────────────────
print("\n[STEP 6] Running Sample Prediction...")
from prediction import predict_single, predict_batch, print_result

sample_patient = {
    'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}
result = predict_single(sample_patient)
print_result(result, sample_patient)

predict_batch('data/heart.csv')

# ── FINAL SUMMARY ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PIPELINE COMPLETE — SUMMARY")
print("=" * 55)
print(f"  Logistic Regression Accuracy : {acc_lr}%")
print(f"  Random Forest Accuracy       : {acc_rf}%")
print(f"  Winner Model                 : Random Forest ✓")
print(f"\n  Output files generated:")
for f in sorted(os.listdir('outputs')):
    print(f"    outputs/{f}")
for f in sorted(os.listdir('models')):
    print(f"    models/{f}")
print("=" * 55)
print("\n  Open index.html in a browser to see the dashboard.")
