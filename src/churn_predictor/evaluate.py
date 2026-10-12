"""Model evaluation visualizations — confusion matrices, ROC curves, SHAP."""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def plot_confusion_matrices(models: dict, X_test, y_test):
    """Plot confusion matrices for all models side by side."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Retained", "Churned"],
                    yticklabels=["Retained", "Churned"])
        ax.set_title(f"{name}\nConfusion Matrix")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    save_path = config.REPORTS_DIR / "confusion_matrices.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrices to {save_path}")


def plot_roc_curves(models: dict, X_test, y_test):
    """Plot overlaid ROC curves for all models."""
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    save_path = config.REPORTS_DIR / "roc_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves to {save_path}")


def plot_feature_importance(model, feature_names, model_name="XGBoost", top_n=15):
    """Plot top-N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title(f"{model_name} — Top {top_n} Features")
    plt.tight_layout()

    save_path = config.REPORTS_DIR / "feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance to {save_path}")


def plot_shap_summary(model, X_test, feature_names):
    """Generate SHAP summary plot for the XGBoost model."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

    save_path = config.REPORTS_DIR / "shap_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary to {save_path}")


def generate_all_plots(models: dict, tuned_xgb, X_test, y_test):
    """Generate all evaluation plots (ROC curves, confusion matrix, feature importance, SHAP)."""
    feature_names = list(X_test.columns)

    # Plot overlaid ROC curves for all models
    plot_roc_curves(models, X_test, y_test)

    # Plot confusion matrix for the best model (Tuned XGBoost)
    plot_confusion_matrices({"Tuned XGBoost (Optuna)": tuned_xgb}, X_test, y_test)

    # Feature importance and SHAP analysis on Tuned XGBoost
    plot_feature_importance(tuned_xgb, feature_names, "Tuned XGBoost (Optuna)")
    plot_shap_summary(tuned_xgb, X_test, feature_names)

    logger.info("All evaluation plots generated successfully!")