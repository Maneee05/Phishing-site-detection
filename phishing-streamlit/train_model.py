# -*- coding: utf-8 -*-
"""
train_model.py
──────────────
Trains a phishing-site detection model on the Kaggle Phishing Websites Dataset
(phishing.csv). Saves the best model as `phishing_model.pkl` and the ordered
feature list as `feature_names.json`.

Run:
    python train_model.py
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "phishing.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "phishing_model.pkl")
FEAT_PATH    = os.path.join(BASE_DIR, "feature_names.json")
PLOTS_DIR    = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Phishing Site Detection — Model Training")
print("=" * 60)
print(f"\n[1/6] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"      Shape : {df.shape}")
print(f"      Columns: {list(df.columns)}")
print(f"\n      Class distribution:\n{df['class'].value_counts()}\n")

# ── 2. Pre-process ────────────────────────────────────────────────────────────
print("[2/6] Pre-processing ...")

# Drop 'Index' column if present
drop_cols = [c for c in ["Index", "index"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Encode target: phishing=1, legitimate=-1 -> remap to 1/0
if df["class"].isin([-1, 1]).all():
    df["class"] = df["class"].map({1: 1, -1: 0})   # 1=phishing, 0=safe

# Handle any remaining -1 categorical values -> keep as-is (model handles them)
# Check nulls
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print(f"      Found {null_counts.sum()} null values — filling with column medians.")
    df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=["class"])
y = df["class"]

feature_names = list(X.columns)
print(f"      Features ({len(feature_names)}): {feature_names}")

# Save feature names
with open(FEAT_PATH, "w") as f:
    json.dump(feature_names, f)
print(f"      Saved feature names -> {FEAT_PATH}")

# ── 3. Split ──────────────────────────────────────────────────────────────────
print("\n[3/6] Splitting data (80/20 stratified) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 4. Train Ensemble ─────────────────────────────────────────────────────────
print("\n[4/6] Training ensemble model (RF + GB VotingClassifier) ...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft",
    n_jobs=-1
)

ensemble.fit(X_train, y_train)
print("      Training complete.")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n[5/6] Evaluating ...")
y_pred      = ensemble.predict(X_test)
y_proba     = ensemble.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

print(f"\n  ┌{'─'*40}┐")
print(f"  │  {'Metric':<20}{'Value':>18}  │")
print(f"  ├{'─'*40}┤")
print(f"  │  {'Accuracy':<20}{acc*100:>17.2f}%  │")
print(f"  │  {'Precision':<20}{prec*100:>17.2f}%  │")
print(f"  │  {'Recall':<20}{rec*100:>17.2f}%  │")
print(f"  │  {'F1 Score':<20}{f1*100:>17.2f}%  │")
print(f"  │  {'ROC-AUC':<20}{auc*100:>17.2f}%  │")
print(f"  └{'─'*40}┘")
print(f"\n{classification_report(y_test, y_pred, target_names=['Safe','Phishing'])}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Safe", "Phishing"],
            yticklabels=["Safe", "Phishing"])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Feature Importances from the RF sub-model
rf_trained = ensemble.estimators_[0]
imp = pd.Series(rf_trained.feature_importances_, index=feature_names).nlargest(15)
imp.sort_values().plot(kind="barh", ax=axes[1], color="#4C72B0")
axes[1].set_title("Top-15 Feature Importances (Random Forest)")
axes[1].set_xlabel("Importance")

plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "evaluation.png")
plt.savefig(cm_path, dpi=150)
print(f"\n  Saved evaluation plot -> {cm_path}")
plt.close()

# ── 6. Save Model ─────────────────────────────────────────────────────────────
print(f"\n[6/6] Saving model -> {MODEL_PATH}")
joblib.dump(ensemble, MODEL_PATH, compress=3)
size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
print(f"      File size: {size_mb:.2f} MB")
print("\n  ✅ Done! Run `streamlit run app.py` to launch the detector.\n")
