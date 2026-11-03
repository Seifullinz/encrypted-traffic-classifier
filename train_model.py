#!/usr/bin/env python3
"""
train_model.py
=====================================================================
Trains a Random Forest on the extracted features, evaluates it, and saves
the model together with the result graphics.

The evaluation mode is selected automatically from the 'source_file' column:
  * Session CSV (one row per pcap)       -> ordinary stratified train/test split
                                            plus StratifiedKFold cross-validation.
  * Windowed CSV (many windows per pcap) -> GROUP-AWARE split: the windows of a
    single capture never appear in both the training and the test set
    (GroupShuffleSplit + StratifiedGroupKFold). This removes data leakage.

Usage:
    python3 train_model.py --features features.csv     --outdir results
    python3 train_model.py --features features_win.csv --outdir results_win
=====================================================================
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI; just save PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_score, train_test_split)

RANDOM_STATE = 42
NON_FEATURE_COLS = ("filename", "label", "source_file")


def load_data(features_csv):
    df = pd.read_csv(features_csv)
    if "label" not in df.columns:
        raise ValueError("The CSV has no 'label' column.")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    # to_numpy(dtype=...) forces a plain numpy array. This matters on pandas 3.x /
    # PyArrow, where .values can return an Arrow array that scikit-learn cannot index.
    X = (df[feature_cols].apply(pd.to_numeric, errors="coerce")
         .fillna(0.0).to_numpy(dtype=float))
    y = df["label"].astype(str).to_numpy(dtype=object)
    if "source_file" in df.columns:
        groups = df["source_file"].astype(str).to_numpy(dtype=object)
    elif "filename" in df.columns:
        groups = df["filename"].astype(str).to_numpy(dtype=object)
    else:
        groups = np.arange(len(y)).astype(object)
    return X, y, groups, feature_cols


def main():
    ap = argparse.ArgumentParser(description="Train and evaluate the Random Forest.")
    ap.add_argument("--features", default="features.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--test-size", type=float, default=0.30)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    X, y, groups, feature_cols = load_data(args.features)
    classes = sorted(np.unique(y))
    n_groups = len(np.unique(groups))
    group_mode = n_groups < len(y)  # windows -> several rows per capture
    mode_txt = (f"GROUP-AWARE (windows; {n_groups} source captures)"
                if group_mode else "standard stratified")

    print(f"Samples: {len(y)} | Features: {len(feature_cols)} | Classes: {classes}")
    for c in classes:
        print(f"  {c}: {(y == c).sum()}")
    print(f"Evaluation mode: {mode_txt}")

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)

    # ---- Hold-out split ----
    if group_mode:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                random_state=RANDOM_STATE)
        tr_idx, te_idx = next(gss.split(X, y, groups))
    else:
        tr_idx, te_idx = train_test_split(
            np.arange(len(y)), test_size=args.test_size,
            stratify=y, random_state=RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, digits=3)

    # ---- Cross-validation ----
    if group_mode:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="accuracy")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # ---- Save the model bundle (used by the dashboard) ----
    joblib.dump({"model": model, "feature_cols": feature_cols, "classes": classes},
                os.path.join(args.outdir, "model.joblib"))

    # ---- metrics.txt ----
    metrics_path = os.path.join(args.outdir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("ENCRYPTED TRAFFIC CLASSIFICATION - RANDOM FOREST\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Evaluation mode: {mode_txt}\n")
        f.write(f"Samples: {len(y)} | Features: {len(feature_cols)}\n\n")
        f.write(f"Hold-out test accuracy: {acc:.4f}\n\n")
        f.write("Classification report (hold-out test set):\n")
        f.write(report + "\n\n")
        f.write("5-fold cross-validation accuracy:\n")
        f.write(f"  per fold: {np.round(cv_scores, 4).tolist()}\n")
        f.write(f"  mean: {cv_scores.mean():.4f}  std: {cv_scores.std():.4f}\n")
    with open(metrics_path, encoding="utf-8") as f:
        print("\n" + f.read())

    # ---- Plot 1: confusion matrix ----
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix (Hold-out Test Set)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 2: feature importance (top 15) ----
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    top.plot(kind="barh", ax=ax, color="#2C5F8A")
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "feature_importance.png"), dpi=150)
    plt.close(fig)

    # ---- Plot 3: cross-validation accuracy per fold ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(cv_scores) + 1), cv_scores, color="#3C8A6E")
    ax.axhline(cv_scores.mean(), color="#B0413E", linestyle="--",
               label=f"mean = {cv_scores.mean():.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("5-Fold Cross-Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "cv_scores.png"), dpi=150)
    plt.close(fig)

    print(f"Model and graphics saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
