import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

FEATURES = ["temp", "vibration", "current", "rpm"]
TARGET   = "label"
TEST_SIZE   = 0.20
RANDOM_SEED = 42


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        print("Run:  python data/synthetic_data.py  first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Class distribution:\n{df[TARGET].value_counts().to_string()}\n")

    X = df[FEATURES].values
    y = df[TARGET].values

    # ── Train / test split (stratified) ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Build pipeline ────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])

    # ── Fit ───────────────────────────────────────────────────────────────────
    pipeline.fit(X_train, y_train)
    print("Training complete.\n")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    y_pred   = pipeline.predict(X_test)
    y_proba  = pipeline.predict_proba(X_test)[:, 1]
    auc_roc  = roc_auc_score(y_test, y_proba)

    print("=" * 55)
    print("CLASSIFICATION REPORT (test set)")
    print("=" * 55)
    print(classification_report(
        y_test, y_pred,
        target_names=["NORMAL", "FAILURE"],
        digits=4,
    ))

    print(f"AUC-ROC (test)   : {auc_roc:.4f}")
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print()

    # ── 5-fold Stratified CV AUC ──────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"5-Fold CV AUC    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Per fold       : {np.round(cv_scores, 4).tolist()}")
    print()

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")
    print(f"File size  : {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
