"""
models.py — Regression model factories and evaluation utilities.

Available models
----------------
  get_xgboost()   — XGBRegressor  (gradient boosting, bonus)
  get_lgbm()      — LGBMRegressor (Light GBM, bonus)

Evaluation helpers
------------------
  calc_metrics()           — RMSE, MAE, R², MAPE for a single set
  evaluate_model()         — print metrics for train / val / test
  plot_actual_vs_pred()    — time-series overlay of actual vs predicted
  plot_feature_importance()— horizontal bar chart
  plot_residuals()         — 2×2 diagnostic panel
  compare_models()         — side-by-side RMSE / MAE / R² bar charts
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.config import RANDOM_STATE, PRIMARY_COLOR, SECONDARY_COLOR


# ─────────────────────────────────────────────────────────────────────────────
# Model factories
# ─────────────────────────────────────────────────────────────────────────────

def get_xgboost(
    n_estimators: int   = 500,
    learning_rate: float = 0.05,
    max_depth: int       = 6,
    subsample: float     = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
) -> XGBRegressor:
    """Return a configured XGBRegressor."""
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def get_lgbm(
    n_estimators: int    = 500,
    learning_rate: float = 0.05,
    max_depth: int       = 6,
    num_leaves: int      = 63,
    subsample: float     = 0.8,
    colsample_bytree: float = 0.8,
):
    """Return a configured LGBMRegressor."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def calc_metrics(y_true, y_pred, set_name: str = "") -> dict:
    """Compute RMSE, MAE, R², and MAPE for a single set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {"Set": set_name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE(%)": mape}


def evaluate_model(
    model,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    model_name: str = "Model",
) -> pd.DataFrame:
    """
    Evaluate on train / val / test and print a formatted metrics table.

    Returns
    -------
    pd.DataFrame with columns [Set, RMSE, MAE, R2, MAPE(%)]
    """
    sets = [
        (X_train, y_train, "Train"),
        (X_val,   y_val,   "Val"),
        (X_test,  y_test,  "Test"),
    ]
    rows = [calc_metrics(y, model.predict(X), name) for X, y, name in sets]
    df_m = pd.DataFrame(rows)

    print(f"\n{'═' * 60}")
    print(f"  {model_name}  —  Evaluation")
    print(f"{'═' * 60}")
    print(df_m.to_string(index=False, float_format="%.4f"))
    return df_m


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_pred(
    dates,
    y_actual,
    predictions: dict,
    title: str  = "Actual vs Predicted — Total Weekly Sales",
    figsize: tuple = (14, 5),
):
    """
    Overlay actual sales and multiple model predictions over time.

    Parameters
    ----------
    dates       : array-like of datetime values (test set)
    y_actual    : array-like of actual sales
    predictions : {model_name: y_pred_array}
    """
    line_colors = [PRIMARY_COLOR, SECONDARY_COLOR, "mediumseagreen", "mediumpurple"]

    plt.figure(figsize=figsize)
    plt.plot(dates, y_actual, label="Actual", color="black", linewidth=2)
    for (name, pred), color in zip(predictions.items(), line_colors):
        plt.plot(dates, pred, label=name, color=color, linewidth=1.8, alpha=0.8)

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Weekly Sales ($)", fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int    = 20,
    title: str    = "Feature Importance",
    figsize: tuple = (10, 7),
):
    """Horizontal bar chart of the top-N feature importances."""
    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1][:top_n]
    names       = [feature_names[i] for i in idx][::-1]
    values      = importances[idx][::-1]
    colors      = [PRIMARY_COLOR if v >= values.mean() else "lightsteelblue" for v in values]

    plt.figure(figsize=figsize)
    plt.barh(names, values, color=colors)
    plt.xlabel("Importance", fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true, y_pred,
    title_prefix: str  = "Model",
    figsize: tuple     = (14, 10),
):
    """2×2 residual diagnostic panel (scatter, histogram, pred-vs-actual, Q-Q)."""
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals vs index
    axes[0, 0].scatter(range(len(residuals)), residuals,
                       alpha=0.4, s=10, color=PRIMARY_COLOR)
    axes[0, 0].axhline(0, color="red", linestyle="--")
    axes[0, 0].set_title(f"{title_prefix} — Residuals vs Index")
    axes[0, 0].set_xlabel("Sample Index")
    axes[0, 0].set_ylabel("Residual ($)")

    # Residuals histogram
    axes[0, 1].hist(residuals, bins=60, edgecolor="black",
                    color=PRIMARY_COLOR, alpha=0.7)
    axes[0, 1].axvline(0, color="red", linestyle="--")
    axes[0, 1].set_title(f"{title_prefix} — Residuals Distribution")
    axes[0, 1].set_xlabel("Residual ($)")
    axes[0, 1].set_ylabel("Frequency")

    # Predicted vs Actual
    axes[1, 0].scatter(y_pred, y_true, alpha=0.3, s=10, color=SECONDARY_COLOR)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    axes[1, 0].plot([lo, hi], [lo, hi], "r--", lw=2)
    axes[1, 0].set_title(f"{title_prefix} — Predicted vs Actual")
    axes[1, 0].set_xlabel("Predicted ($)")
    axes[1, 0].set_ylabel("Actual ($)")

    # Q-Q plot
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f"{title_prefix} — Q-Q Plot")
    except ImportError:
        axes[1, 1].text(0.5, 0.5, "scipy not installed\npip install scipy",
                        ha="center", va="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()
    plt.show()


def compare_models(
    metrics_list: list,
    figsize: tuple = (15, 5),
):
    """
    Side-by-side bar charts comparing RMSE, MAE, R² across models.

    Parameters
    ----------
    metrics_list : list of dicts with keys: model, val_rmse, val_mae, val_r2
    """
    df      = pd.DataFrame(metrics_list)
    colors  = [PRIMARY_COLOR, SECONDARY_COLOR, "mediumseagreen", "mediumpurple"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (col, label) in zip(axes, [
        ("val_rmse", "Validation RMSE  (↓ better)"),
        ("val_mae",  "Validation MAE   (↓ better)"),
        ("val_r2",   "Validation R²    (↑ better)"),
    ]):
        bars = ax.bar(df["model"], df[col], color=colors[:len(df)])
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, df[col]):
            fmt = f"{val:.4f}" if "r2" in col else f"${val:,.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, fmt,
                    ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()
