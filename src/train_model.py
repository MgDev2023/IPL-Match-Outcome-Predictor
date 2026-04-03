"""
Train Logistic Regression + Random Forest on engineered IPL features.
Saves models and evaluation plots to models/ directory.
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import run as build_features_run

FEATURE_COLS = [
    "team1_won_toss",
    "toss_bat",
    "team1_home",
    "team2_home",
    "team1_form5",
    "team2_form5",
    "form_diff",
    "h2h_team1_win_pct",
    "toss_venue_adv",
    "season",
]

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_data():
    feat_path = Path("data/processed/features.csv")
    if not feat_path.exists():
        print("Feature file not found — running feature engineering first...")
        build_features_run()
    df = pd.read_csv(feat_path)
    # Chronological train/test split (last 2 seasons = test)
    df["date"] = pd.to_datetime(df["date"])
    split_date = df["date"].quantile(0.8)
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test


def get_xy(df):
    X = df[FEATURE_COLS].fillna(0.5)
    y = df["target"]
    return X, y


def evaluate(name, model, X_test, y_test, ax_roc):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*40}")
    print(f"{name}  |  AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["team2 wins", "team1 wins"]))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    return auc


def plot_feature_importance(lr_pipeline, rf_pipeline):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")

    # Logistic Regression coefficients
    lr = lr_pipeline.named_steps["clf"]
    coefs = pd.Series(lr.coef_[0], index=FEATURE_COLS).sort_values()
    colors = ["#d73027" if c < 0 else "#4575b4" for c in coefs]
    axes[0].barh(coefs.index, coefs.values, color=colors)
    axes[0].set_title("Logistic Regression Coefficients")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Coefficient value")

    # Random Forest importances
    rf = rf_pipeline.named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    axes[1].barh(imp.index, imp.values, color="#4575b4")
    axes[1].set_title("Random Forest Feature Importance")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    fig.savefig(MODELS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: models/feature_importance.png")


def plot_confusion(model, name, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["team2 wins", "team1 wins"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    fname = f"confusion_{name.lower().replace(' ', '_')}.png"
    fig.savefig(MODELS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: models/{fname}")


def run():
    print("Loading data...")
    train, test = load_data()
    X_train, y_train = get_xy(train)
    X_test, y_test = get_xy(test)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Models ──────────────────────────────────────────
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
    ])

    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("\nCross-validating...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, pipe in [("Logistic Regression", lr_pipeline), ("Random Forest", rf_pipeline)]:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"  {name}: CV AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    print("\nFitting on full training set...")
    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # ── Evaluation plots ─────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    lr_auc = evaluate("Logistic Regression", lr_pipeline, X_test, y_test, ax_roc)
    rf_auc = evaluate("Random Forest", rf_pipeline, X_test, y_test, ax_roc)

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve — Test Set")
    ax_roc.legend()
    fig_roc.savefig(MODELS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: models/roc_curve.png")

    plot_confusion(lr_pipeline, "Logistic Regression", X_test, y_test)
    plot_confusion(rf_pipeline, "Random Forest", X_test, y_test)
    plot_feature_importance(lr_pipeline, rf_pipeline)

    # ── Save models ───────────────────────────────────────
    joblib.dump(lr_pipeline, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest.joblib")
    print("\nModels saved to models/")

    # Save best model name for the app
    best = "random_forest" if rf_auc >= lr_auc else "logistic_regression"
    (MODELS_DIR / "best_model.txt").write_text(best)
    print(f"Best model: {best}")

    return lr_pipeline, rf_pipeline


if __name__ == "__main__":
    run()
