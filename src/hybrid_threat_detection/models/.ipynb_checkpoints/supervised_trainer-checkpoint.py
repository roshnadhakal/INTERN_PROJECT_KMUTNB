import os
import pickle
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from ..config.constants import RESULTS_DIR
from ..utils.helpers import (
    create_dir_if_not_exists, 
    save_json, 
    setup_logging
)

logger = logging.getLogger(__name__)

class SupervisedModelTrainer:
    """Production-grade supervised model training for threat detection."""

    def __init__(self, config_path: str = None):
        setup_logging()
        self.models = {}
        self.best_params = {}
        self.evaluation_results = {}

        # Create output directories
        self.model_dir = os.path.join(RESULTS_DIR, "phase3", "models")
        self.result_dir = os.path.join(RESULTS_DIR, "phase3", "results")
        create_dir_if_not_exists(self.model_dir)
        create_dir_if_not_exists(self.result_dir)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize models with default parameters
        self._init_models()

    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from JSON file."""
        default_config = {
            "models": {
                "random_forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "random_state": 42
                },
                "xgboost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1],
                    "random_state": 42
                },
                "logistic_regression": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                    "max_iter": [1000]
                },
                "svm": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                }
            },
            "grid_search": {
                "cv": 5,
                "scoring": "f1",
                "n_jobs": -1,
                "verbose": 1
            },
            "datasets": ["xss", "sql_injection", "path_traversal"]
        }

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded model config from {config_path}")
            return config
        else:
            logger.info("Using default model configuration")
            return default_config

    def _init_models(self) -> None:
        """Initialize model instances with default parameters."""
        self.base_models = {
            "random_forest": RandomForestClassifier(),
            "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            "logistic_regression": LogisticRegression(),
            "svm": SVC(probability=True)
        }

    def load_data(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load vectorized data and splits for a dataset."""
        try:
            train_df = pd.read_csv(f"data/splits/{dataset_name}/train.csv")
            test_df = pd.read_csv(f"data/splits/{dataset_name}/test.csv")
            vectorizer_path = f"src/hybrid_threat_detection/results/phase2/vectorizers/{dataset_name}_tfidf.pkl"
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)

            column_name = "payload"
            if dataset_name == "xss":
                column_name = "post_data"

            if column_name not in train_df.columns or column_name not in test_df.columns:
                raise KeyError(f"Missing '{column_name}' column in data")

            X_train = vectorizer.transform(train_df[column_name].astype(str))
            X_test = vectorizer.transform(test_df[column_name].astype(str))
            y_train = train_df["label"].values
            y_test = test_df["label"].values

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error loading data for {dataset_name}: {str(e)}")
            raise

    def train_models(self) -> Dict[str, Any]:
        results = {}

        for dataset in self.config["datasets"]:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training models on {dataset} dataset")
                logger.info(f"{'='*50}")

                X_train, X_test, y_train, y_test = self.load_data(dataset)
                dataset_results = {}

                for model_name, model in self.base_models.items():
                    logger.info(f"\nTraining {model_name} on {dataset}")

                    try:
                        best_model, best_params = self._train_with_gridsearch(
                            model_name, model, X_train, y_train
                        )
                        metrics = self._evaluate_model(best_model, X_test, y_test)
                        dataset_results[model_name] = {
                            "best_params": best_params,
                            "metrics": metrics,
                            "model_path": self._save_model(best_model, dataset, model_name)
                        }

                    except Exception as model_err:
                        logger.error(f"Model {model_name} failed on {dataset}: {str(model_err)}")
                        continue

                results[dataset] = dataset_results
                self._save_dataset_results(dataset, dataset_results)

            except Exception as e:
                logger.error(f"Error processing dataset {dataset}: {str(e)}")
                continue

        self.evaluation_results = results
        self._save_combined_results()
        return results

    def _train_with_gridsearch(self, model_name: str, model: Any, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Dict]:
        params = self.config["models"].get(model_name, {})
        grid_params = {k: v for k, v in params.items() if k != "random_state"}

        if not grid_params:
            logger.info(f"No hyperparameters to tune for {model_name}. Using default.")
            model.fit(X, y)
            return model, model.get_params()

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            **self.config["grid_search"]
        )

        logger.info(f"Starting GridSearchCV for {model_name}")
        grid_search.fit(X, y)
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

        return grid_search.best_estimator_, grid_search.best_params_

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred,zero_division=0),
            "recall": recall_score(y, y_pred,zero_division=0),
            "f1": f1_score(y, y_pred,zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }

        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y, y_prob)
            except Exception:
                metrics["roc_auc"] = None

        logger.info(f"Evaluation metrics:\n{pd.DataFrame([metrics])}")
        return metrics

    def _save_model(self, model: Any, dataset: str, model_name: str) -> str:
        model_dir = os.path.join(self.model_dir, dataset)
        create_dir_if_not_exists(model_dir)
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} model to {model_path}")
        return model_path

    def _save_dataset_results(self, dataset: str, results: Dict) -> None:
        result_dir = os.path.join(self.result_dir, dataset)
        create_dir_if_not_exists(result_dir)
        metrics_path = os.path.join(result_dir, "metrics.json")
        save_json(results, metrics_path)
        self._plot_confusion_matrices(dataset, results, result_dir)

    def _plot_confusion_matrices(self, dataset: str, results: Dict, output_dir: str) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        for model_name, result in results.items():
            cm = result["metrics"]["confusion_matrix"]
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Benign", "Malicious"],
                        yticklabels=["Benign", "Malicious"])
            plt.title(f"Confusion Matrix - {model_name} ({dataset})")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plot_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved confusion matrix to {plot_path}")

    def _save_combined_results(self) -> None:
        combined_path = os.path.join(self.result_dir, "combined_results.json")
        save_json(self.evaluation_results, combined_path)
        summary = {}
        for dataset, models in self.evaluation_results.items():
            summary[dataset] = {
                model_name: {
                    "best_params": model_info["best_params"],
                    "metrics": {k: v for k, v in model_info["metrics"].items() if k != "confusion_matrix"}
                } for model_name, model_info in models.items()
            }
        summary_path = os.path.join(self.result_dir, "summary_report.json")
        save_json(summary, summary_path)
        logger.info(f"Saved combined results to {combined_path}")
        logger.info(f"Saved summary report to {summary_path}")

    def get_results(self) -> Dict:
        return self.evaluation_results
