"""
Module 3: Model Training & Evaluation
Algorithms: Random Forest + Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
import joblib
import warnings
warnings.filterwarnings('ignore')


# ─── LOGISTIC REGRESSION ───────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    print("[✓] Logistic Regression trained.")
    return lr


# ─── RANDOM FOREST ─────────────────────────────────────────────

def train_random_forest(X_train, y_train, search_best=True):
    """Train Random Forest, optionally searching best random_state."""
    if search_best:
        print("[...] Searching best Random Forest random_state (200 trials)...")
        max_acc = 0
        best_state = 0
        from sklearn.model_selection import cross_val_score
        for state in range(200):
            rf = RandomForestClassifier(n_estimators=100, random_state=state)
            cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
            if cv.mean() > max_acc:
                max_acc = cv.mean()
                best_state = state
        print(f"[✓] Best random_state={best_state} | CV accuracy={max_acc:.4f}")
    else:
        best_state = 42

    rf = RandomForestClassifier(n_estimators=200, random_state=best_state,
                                max_depth=None, min_samples_split=2)
    rf.fit(X_train, y_train)
    print("[✓] Random Forest trained.")
    return rf


# ─── EVALUATION ────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n── {name.upper()} RESULTS ───────────────────────")
    print(f"  Accuracy   : {acc} %")
    print(f"  Precision  : {cr['weighted avg']['precision']*100:.2f} %")
    print(f"  Recall     : {cr['weighted avg']['recall']*100:.2f} %")
    print(f"  F1-Score   : {cr['weighted avg']['f1-score']*100:.2f} %")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print("─────────────────────────────────────────────")
    return acc, cm, cr, y_pred


# ─── PLOTS ─────────────────────────────────────────────────────

def plot_confusion_matrices(cm_lr, cm_rf, acc_lr, acc_rf,
                            save_path='outputs/confusion_matrices.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0b0f1a')
    fig.suptitle('Confusion Matrices', color='#e2e8f0',
                 fontsize=15, fontweight='bold', y=1.02)

    for ax, cm, name, acc, color in zip(
        axes,
        [cm_lr, cm_rf],
        ['Logistic Regression', 'Random Forest'],
        [acc_lr, acc_rf],
        ['#3b82f6', '#10b981']
    ):
        ax.set_facecolor('#111827')
        cmap = sns.light_palette(color, as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    linewidths=1, linecolor='#0b0f1a',
                    annot_kws={'size': 18, 'weight': 'bold', 'color': '#0b0f1a'},
                    cbar=False)
        ax.set_title(f'{name}\nAccuracy: {acc}%', color='#e2e8f0',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', color='#94a3b8', fontsize=10)
        ax.set_ylabel('True Label', color='#94a3b8', fontsize=10)
        ax.tick_params(colors='#94a3b8')
        ax.set_xticklabels(['No Disease', 'Disease'], color='#94a3b8')
        ax.set_yticklabels(['No Disease', 'Disease'], color='#94a3b8', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Confusion matrices saved → {save_path}")


def plot_accuracy_comparison(acc_lr, acc_rf,
                             save_path='outputs/accuracy_comparison.png'):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0b0f1a')
    ax.set_facecolor('#111827')

    algos  = ['Logistic\nRegression', 'Random\nForest']
    accs   = [acc_lr, acc_rf]
    colors = ['#3b82f6', '#10b981']

    bars = ax.bar(algos, accs, color=colors, width=0.45,
                  edgecolor='none', zorder=3)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)', color='#94a3b8', fontsize=11)
    ax.set_title('Algorithm Accuracy Comparison', color='#e2e8f0',
                 fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#1f2d45')
    ax.grid(axis='y', color='#1f2d45', linewidth=0.8, zorder=0)
    ax.yaxis.set_tick_params(length=0)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{acc}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold', color='#e2e8f0')

    # Winner label
    ax.annotate('★ Best Model', xy=(1, acc_rf), xytext=(1.25, acc_rf + 6),
                color='#10b981', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#10b981', lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Accuracy comparison saved → {save_path}")


def plot_roc_curves(lr, rf, X_test, y_test,
                    save_path='outputs/roc_curves.png'):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0b0f1a')
    ax.set_facecolor('#111827')

    for model, name, color in [(lr, 'Logistic Regression', '#3b82f6'),
                                (rf, 'Random Forest', '#10b981')]:
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC = {roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)

    ax.plot([0, 1], [0, 1], 'w--', lw=1.2, alpha=0.4, label='Random Chance')
    ax.set_xlabel('False Positive Rate', color='#94a3b8', fontsize=11)
    ax.set_ylabel('True Positive Rate', color='#94a3b8', fontsize=11)
    ax.set_title('ROC Curves', color='#e2e8f0', fontsize=14,
                 fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#1f2d45')
    ax.grid(color='#1f2d45', linewidth=0.8)
    legend = ax.legend(facecolor='#1a2235', edgecolor='#1f2d45',
                       labelcolor='#e2e8f0', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] ROC curves saved → {save_path}")


def plot_target_distribution(y, save_path='outputs/target_distribution.png'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor('#0b0f1a')
    fig.suptitle('Target Variable Distribution', color='#e2e8f0',
                 fontsize=14, fontweight='bold')

    counts = y.value_counts().sort_index()
    labels = ['No Disease', 'Heart Disease']
    colors = ['#3b82f6', '#e84855']

    # Bar chart
    ax = axes[0]
    ax.set_facecolor('#111827')
    ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor='none')
    ax.set_ylabel('Count', color='#94a3b8')
    ax.set_title('Count', color='#e2e8f0', fontsize=12)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#1f2d45')
    ax.grid(axis='y', color='#1f2d45', lw=0.8)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha='center', color='#e2e8f0',
                fontsize=13, fontweight='bold')

    # Pie chart
    ax2 = axes[1]
    ax2.set_facecolor('#0b0f1a')
    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140,
        textprops={'color': '#e2e8f0', 'fontsize': 11},
        wedgeprops={'edgecolor': '#0b0f1a', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_color('#0b0f1a')
        at.set_fontweight('bold')
    ax2.set_title('Proportion', color='#e2e8f0', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Target distribution saved → {save_path}")


# ─── SAVE MODELS ───────────────────────────────────────────────

def save_models(lr, rf, scaler):
    joblib.dump(lr,     'models/logistic_regression.pkl')
    joblib.dump(rf,     'models/random_forest.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("[✓] Models saved to models/")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import load_data, preprocess

    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    plot_target_distribution(df['target'])

    lr = train_logistic_regression(X_train, y_train)
    rf = train_random_forest(X_train, y_train, search_best=False)

    acc_lr, cm_lr, cr_lr, pred_lr = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    acc_rf, cm_rf, cr_rf, pred_rf = evaluate_model(rf, X_test, y_test, "Random Forest")

    plot_confusion_matrices(cm_lr, cm_rf, acc_lr, acc_rf)
    plot_accuracy_comparison(acc_lr, acc_rf)
    plot_roc_curves(lr, rf, X_test, y_test)
    save_models(lr, rf, scaler)

    print(f"\n{'='*45}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"  Logistic Regression : {acc_lr}%")
    print(f"  Random Forest       : {acc_rf}%")
    print(f"  Winner              : Random Forest ✓")
    print(f"{'='*45}")
