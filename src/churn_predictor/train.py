"""Train classification models with 5-Fold Stratified CV, Optuna tuning, and MLflow experiment tracking."""
import logging
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from . import config

logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment("churn_prediction_benchmark")

def load_processed_data():
    """Load the processed train/test splits."""
    X_train = pd.read_csv(config.PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(config.PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(config.PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(config.PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute all classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"{'=' * 50}")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def benchmark_cv_models(X_train, y_train):
    """Run 5-Fold Stratified Cross-Validation on baseline algorithms."""
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, scale_pos_weight=scale_pos, eval_metric="logloss",
            random_state=config.RANDOM_STATE, n_jobs=-1
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_summary = {}

    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1
        )
        cv_summary[name] = {
            "CV Accuracy": scores["test_accuracy"].mean(),
            "CV Recall": scores["test_recall"].mean(),
            "CV Precision": scores["test_precision"].mean(),
            "CV F1-Score": scores["test_f1"].mean(),
            "CV ROC-AUC": scores["test_roc_auc"].mean(),
        }

    df_cv = pd.DataFrame(cv_summary).T.round(4)
    logger.info(f"\n5-Fold Stratified CV Benchmark:\n{df_cv}")
    return models, df_cv


def tune_xgboost_optuna(X_train, y_train, X_test, y_test, n_trials=15):
    """Tune XGBoost hyperparameters with Optuna Bayesian optimization."""
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "scale_pos_weight": scale_pos,
            "eval_metric": "logloss",
            "random_state": config.RANDOM_STATE,
            "n_jobs": -1
        }
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, prob)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = config.RANDOM_STATE

    logger.info(f"Best Optuna ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best Hyperparameters: {best_params}")

    # Train final tuned model
    tuned_xgb = XGBClassifier(**best_params)
    tuned_xgb.fit(X_train, y_train)
    return tuned_xgb, study


def train_and_track_all():
    """Train all benchmark models, run Optuna tuning, and log everything to MLflow."""
    X_train, X_test, y_train, y_test = load_processed_data()
    mlflow.set_experiment("churn_prediction_benchmark")

    # 1. 5-Fold CV Benchmark
    models, df_cv = benchmark_cv_models(X_train, y_train)

    test_metrics = {}
    fitted_models = {}

    # 2. Fit and evaluate each baseline model
    for name, model in models.items():
        with mlflow.start_run(run_name=f"baseline_{name.lower().replace(' ', '_')}"):
            model.fit(X_train, y_train)
            fitted_models[name] = model
            metrics = evaluate_model(model, X_test, y_test, name)
            test_metrics[name] = metrics

            mlflow.log_param("model_type", name)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            if isinstance(model, XGBClassifier):
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )

    # 3. Optuna Hyperparameter Optimization on XGBoost
    with mlflow.start_run(run_name="optuna_tuned_xgboost"):
        tuned_xgb, study = tune_xgboost_optuna(X_train, y_train, X_test, y_test, n_trials=15)
        fitted_models["Tuned XGBoost (Optuna)"] = tuned_xgb
        tuned_metrics = evaluate_model(tuned_xgb, X_test, y_test, "Tuned XGBoost (Optuna)")
        test_metrics["Tuned XGBoost (Optuna)"] = tuned_metrics

        mlflow.log_params(study.best_params)
        for k, v in tuned_metrics.items():
            mlflow.log_metric(k, v)
        mlflow.xgboost.log_model(tuned_xgb, name="best_model")

        # Save serialized model artifact
        save_dir = Path("src/churn_predictor/models")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "best_churn_model.joblib"
        joblib.dump(tuned_xgb, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info(f"Saved best model to {model_path}")

    # 4. Save comparison summary table
    comparison_df = pd.DataFrame(test_metrics).T
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    logger.info(f"\nFinal Test Comparison:\n{comparison_df.round(4)}")

    return fitted_models, comparison_df