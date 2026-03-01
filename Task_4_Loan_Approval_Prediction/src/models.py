"""
models.py — Classifier definitions and evaluation utilities.

Available classifiers:
  1. get_logistic_regression()  — LogisticRegression (baseline)
  2. get_decision_tree()        — DecisionTreeClassifier
  3. get_random_forest()        — RandomForestClassifier (ensemble)
  4. get_gradient_boosting()    — GradientBoostingClassifier (bonus)

Evaluation helpers:
  - evaluate_model()            — classification report + ROC-AUC
  - plot_confusion_matrix()     — annotated heatmap
  - plot_roc_curves()           — multi-model ROC overlay
  - plot_precision_recall()     — Precision-Recall curves
  - plot_feature_importance()   — bar chart for tree-based models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score

from src.config import RANDOM_STATE


# ─────────────────────────────────────────────────────────────────────────────
# Classifier factories
# ─────────────────────────────────────────────────────────────────────────────

def get_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight=None,
) -> LogisticRegression:
    """Return a compiled LogisticRegression."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        solver="lbfgs",
    )


def get_decision_tree(
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight=None,
) -> DecisionTreeClassifier:
    """Return a DecisionTreeClassifier."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )


def get_random_forest(
    n_estimators: int = 100,
    max_depth: int | None = None,
    class_weight=None,
) -> RandomForestClassifier:
    """Return a RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        n_jobs=-1,
    )


def get_gradient_boosting(
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
) -> GradientBoostingClassifier:
    """Return a GradientBoostingClassifier."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
) -> dict:
    """
    Print full classification report and return a metrics dict.

    Returns
    -------
    dict with keys: model, accuracy, precision, recall, f1_macro, f1_weighted, roc_auc
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n{'═' * 55}")
    print(f"  {model_name}  —  Evaluation on Test Set")
    print(f"{'═' * 55}")
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

    metrics = {
        "model":        model_name,
        "accuracy":     (y_pred == y_test).mean(),
        "f1_approved":  f1_score(y_test, y_pred, pos_label=1),
        "f1_rejected":  f1_score(y_test, y_pred, pos_label=0),
        "f1_macro":     f1_score(y_test, y_pred, average="macro"),
        "f1_weighted":  f1_score(y_test, y_pred, average="weighted"),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC : {metrics['roc_auc']:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    figsize: tuple = (6, 5),
):
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        linewidths=0.5, cbar=True,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True",      fontsize=12)
    plt.title(title,        fontsize=13)
    plt.tight_layout()
    plt.show()
    return cm


def plot_roc_curves(
    models_dict: dict,
    X_test,
    y_test,
    figsize: tuple = (8, 6),
):
    """Overlay ROC curves for multiple models."""
    plt.figure(figsize=figsize)
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba     = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc         = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, lw=2, label=f"{name}  (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate",  fontsize=12)
    plt.title("ROC Curves — Model Comparison", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall(
    models_dict: dict,
    X_test,
    y_test,
    figsize: tuple = (8, 6),
):
    """Precision-Recall curves for multiple models."""
    plt.figure(figsize=figsize)
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_proba         = model.predict_proba(X_test)[:, 1]
            prec, rec, _    = precision_recall_curve(y_test, y_proba)
            ap              = average_precision_score(y_test, y_proba)
            plt.plot(rec, prec, lw=2, label=f"{name}  (AP = {ap:.3f})")
    plt.xlabel("Recall",    fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    title: str = "Feature Importance",
    figsize: tuple = (10, 5),
):
    """Horizontal bar chart of feature importances for tree-based models."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals  = importances[indices]

    plt.figure(figsize=figsize)
    colors = ["steelblue" if v >= sorted_vals.mean() else "lightsteelblue" for v in sorted_vals]
    plt.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
    plt.xlabel("Importance", fontsize=12)
    plt.title(title,         fontsize=13)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
