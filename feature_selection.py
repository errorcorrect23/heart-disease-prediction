"""
Module 2: Feature Selection
Uses correlation analysis + Random Forest feature importance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


FEATURE_LABELS = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain',
    'trestbps': 'Rest BP', 'chol': 'Cholesterol', 'fbs': 'Fasting BS',
    'restecg': 'Rest ECG', 'thalach': 'Max HR', 'exang': 'Ex Angina',
    'oldpeak': 'ST Depression', 'slope': 'ST Slope',
    'ca': 'Major Vessels', 'thal': 'Thalassemia'
}


def correlation_analysis(df):
    """Correlation of each feature with target."""
    corr = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
    print("\n── CORRELATION WITH TARGET ──────────────────")
    for feat, val in corr.items():
        bar = '█' * int(val * 30)
        print(f"  {feat:<12} {val:.4f}  {bar}")
    print("─────────────────────────────────────────────")
    return corr


def rf_importance(X, y):
    """Random Forest feature importance scores."""
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)
    return importance


def select_top_features(X, y, k=10):
    """Select top-k features using RF importance."""
    imp = rf_importance(X, y)
    top_features = imp.head(k).index.tolist()
    print(f"\n[✓] Top {k} features selected:")
    for i, f in enumerate(top_features, 1):
        print(f"    {i:2}. {f:<12} importance={imp[f]:.4f}")
    return top_features, imp


def plot_feature_importance(importance, save_path='outputs/feature_importance.png'):
    """Bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0b0f1a')
    ax.set_facecolor('#111827')

    colors = ['#e84855' if i < 5 else '#3b82f6' if i < 9 else '#64748b'
              for i in range(len(importance))]
    bars = ax.barh(importance.index[::-1], importance.values[::-1],
                   color=colors[::-1], edgecolor='none', height=0.7)

    ax.set_xlabel('Feature Importance Score', color='#94a3b8', fontsize=11)
    ax.set_title('Random Forest — Feature Importance', color='#e2e8f0',
                 fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color('#1f2d45')
    ax.grid(axis='x', color='#1f2d45', linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, importance.values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='#94a3b8', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Feature importance chart saved → {save_path}")


def plot_correlation_heatmap(df, save_path='outputs/correlation_heatmap.png'):
    """Correlation heatmap."""
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0b0f1a')
    ax.set_facecolor('#111827')

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                fmt='.2f', linewidths=0.5, ax=ax,
                annot_kws={'size': 8, 'color': '#e2e8f0'},
                cbar_kws={'shrink': 0.8})

    ax.set_title('Feature Correlation Matrix', color='#e2e8f0',
                 fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='#94a3b8', labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Correlation heatmap saved → {save_path}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    from preprocessing import load_data, preprocess
    df = load_data()
    corr = correlation_analysis(df)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    top_features, importance = select_top_features(X_train, y_train, k=10)
    plot_feature_importance(importance)
    plot_correlation_heatmap(df)
    print("\n[✓] Feature selection complete.")
