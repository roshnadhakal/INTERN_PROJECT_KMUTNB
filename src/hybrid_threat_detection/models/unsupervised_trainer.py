import os
import pickle
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    homogeneity_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Input,
    Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau
)
from ..config.constants import RESULTS_DIR
from ..utils.helpers import (
    create_dir_if_not_exists,
    save_json,
    setup_logging
)

logger = logging.getLogger(__name__)

class UnsupervisedTrainer:
    """Production-grade unsupervised model training for threat detection."""

    def __init__(self, config_path: str = None):
        setup_logging()
        self.models = {}
        self.evaluation_results = {}

        # Create output directories
        self.model_dir = os.path.join(RESULTS_DIR, "phase3", "unsupervised_models")
        self.result_dir = os.path.join(RESULTS_DIR, "phase3", "unsupervised_results")
        create_dir_if_not_exists(self.model_dir)
        create_dir_if_not_exists(self.result_dir)

        # Load configuration
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict:
        """Load unsupervised configuration from JSON file."""
        default_config = {
            "models": {
                "kmeans": {
                    "n_clusters": 2,
                    "random_state": 42,
                    "n_init": "auto"
                },
                "dbscan": {
                    "eps": 0.5,
                    "min_samples": 5,
                    "metric": "euclidean"
                },
                "autoencoder": {
                    "encoding_dim": 32,
                    "hidden_dims": [64, 32],
                    "dropout_rate": 0.2,
                    "batch_norm": True,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "epochs": 100,
                    "reconstruction_threshold": 0.5,
                    "scale_features": True
                }
            },
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 5,
                    "restore_best_weights": True
                },
                "model_checkpoint": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_best_only": True
                },
                "reduce_lr": {
                    "monitor": "val_loss",
                    "factor": 0.2,
                    "patience": 3,
                    "min_lr": 1e-6
                }
            },
            "datasets": ["xss", "sql_injection", "path_traversal"]
        }

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded unsupervised config from {config_path}")
            return config
        else:
            logger.info("Using default unsupervised configuration")
            return default_config

    def load_data(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load vectorized data for a dataset."""
        try:
            # Load the full dataset (not split for unsupervised)
            df = pd.read_csv(f"src/hybrid_threat_detection/results/phase2/encoding/{dataset_name}_encoded.csv")
            
            # Load vectorizer
            vectorizer_path = f"src/hybrid_threat_detection/results/phase2/vectorizers/{dataset_name}_tfidf.pkl"
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)

            # Get payload column name
            payload_col = "post_data" if dataset_name == "xss" else "payload"
            
            if payload_col not in df.columns:
                raise KeyError(f"Missing '{payload_col}' column in data")

            # Transform texts
            X = vectorizer.transform(df[payload_col].astype(str)).toarray()
            y = df["label"].values if "label" in df.columns else None

            return X, y, df

        except Exception as e:
            logger.error(f"Error loading data for {dataset_name}: {str(e)}")
            raise

    def train_models(self) -> Dict[str, Any]:
        """Train and evaluate all unsupervised models on all datasets."""
        results = {}

        for dataset in self.config["datasets"]:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training unsupervised models on {dataset} dataset")
                logger.info(f"{'='*50}")

                # Load data
                X, y, df = self.load_data(dataset)
                dataset_results = {}

                # Train clustering models
                if self.config["models"].get("kmeans", {}):
                    logger.info("\nTraining KMeans model")
                    kmeans_results = self._train_kmeans(X, y, dataset)
                    dataset_results["kmeans"] = kmeans_results

                if self.config["models"].get("dbscan", {}):
                    logger.info("\nTraining DBSCAN model")
                    dbscan_results = self._train_dbscan(X, y, dataset)
                    dataset_results["dbscan"] = dbscan_results

                # Train autoencoder
                if self.config["models"].get("autoencoder", {}):
                    logger.info("\nTraining Autoencoder model")
                    autoencoder_results = self._train_autoencoder(X, y, dataset, df)
                    dataset_results["autoencoder"] = autoencoder_results

                results[dataset] = dataset_results
                self._save_dataset_results(dataset, dataset_results)

            except Exception as e:
                logger.error(f"Error processing dataset {dataset}: {str(e)}")
                continue

        self.evaluation_results = results
        self._save_combined_results()
        return results

    def _train_kmeans(self, X: np.ndarray, y: np.ndarray, dataset: str) -> Dict:
        """Train and evaluate KMeans clustering."""
        config = self.config["models"]["kmeans"]
        model = KMeans(**config)
        
        # Fit model
        clusters = model.fit_predict(X)
        
        # Evaluate
        metrics = {}
        if y is not None:
            metrics = {
                "adjusted_rand_score": adjusted_rand_score(y, clusters),
                "homogeneity_score": homogeneity_score(y, clusters),
                "silhouette_score": silhouette_score(X, clusters)
            }
            
            # Create confusion matrix (convert clusters to binary labels)
            majority_cluster = np.argmax(np.bincount(clusters))
            pred_labels = (clusters == majority_cluster).astype(int)
            metrics["confusion_matrix"] = confusion_matrix(y, pred_labels).tolist()
        
        # Save model
        model_path = os.path.join(self.model_dir, dataset, "kmeans.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"KMeans metrics: {metrics}")
        return {
            "metrics": metrics,
            "model_path": model_path,
            "n_clusters": config["n_clusters"],
            "cluster_distribution": np.bincount(clusters).tolist()
        }

    def _train_dbscan(self, X: np.ndarray, y: np.ndarray, dataset: str) -> Dict:
        """Train and evaluate DBSCAN clustering."""
        config = self.config["models"]["dbscan"]
        model = DBSCAN(**config)
        
        # Fit model
        clusters = model.fit_predict(X)
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        # Evaluate
        metrics = {}
        if y is not None and n_clusters > 1:  # Need at least 2 clusters to evaluate
            metrics = {
                "adjusted_rand_score": adjusted_rand_score(y, clusters),
                "homogeneity_score": homogeneity_score(y, clusters),
                "silhouette_score": silhouette_score(X, clusters),
                "n_clusters": n_clusters,
                "noise_points": np.sum(clusters == -1)
            }
            
            # Create confusion matrix for binary case
            if n_clusters == 2:
                pred_labels = (clusters == np.argmax(np.bincount(clusters[clusters != -1]))).astype(int)
                pred_labels[clusters == -1] = -1  # Mark noise points
                metrics["confusion_matrix"] = confusion_matrix(y, pred_labels).tolist()
        
        # Save model
        model_path = os.path.join(self.model_dir, dataset, "dbscan.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"DBSCAN metrics: {metrics}")
        return {
            "metrics": metrics,
            "model_path": model_path,
            "cluster_distribution": np.bincount(clusters[clusters != -1]).tolist(),
            "noise_points": np.sum(clusters == -1)
        }

    def _train_autoencoder(self, X: np.ndarray, y: np.ndarray, dataset: str, df: pd.DataFrame) -> Dict:
        """Train and evaluate Autoencoder for anomaly detection."""
        config = self.config["models"]["autoencoder"]
        
        # Scale features if configured
        if config["scale_features"]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Build autoencoder
        input_dim = X_scaled.shape[1]
        encoding_dim = config["encoding_dim"]
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder = input_layer
        
        for dim in config["hidden_dims"]:
            encoder = Dense(dim, activation='relu')(encoder)
            if config["batch_norm"]:
                encoder = BatchNormalization()(encoder)
            encoder = Dropout(config["dropout_rate"])(encoder)
        
        # Bottleneck
        encoder = Dense(encoding_dim, activation='relu')(encoder)
        
        # Decoder (mirror encoder)
        decoder = encoder
        for dim in reversed(config["hidden_dims"]):
            decoder = Dense(dim, activation='relu')(decoder)
            if config["batch_norm"]:
                decoder = BatchNormalization()(decoder)
            decoder = Dropout(config["dropout_rate"])(decoder)
        
        # Output layer
        decoder = Dense(input_dim, activation='linear')(decoder)
        
        # Compile model
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        optimizer = Adam(learning_rate=config["learning_rate"])
        autoencoder.compile(optimizer=optimizer, loss='mse')
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(**self.config["callbacks"]["early_stopping"]),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, dataset, "autoencoder_best.h5"),
                **self.config["callbacks"]["model_checkpoint"]
            ),
            ReduceLROnPlateau(**self.config["callbacks"]["reduce_lr"])
        ]
        
        # Train model
        history = autoencoder.fit(
            X_scaled, X_scaled,
            validation_split=0.2,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate reconstruction errors
        reconstructions = autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        
        # Evaluate performance (if labels available)
        metrics = {}
        if y is not None:
            # Find optimal threshold (you could also use config["reconstruction_threshold"])
            threshold = np.percentile(mse, 95)  # Example: 95th percentile as threshold
            pred_labels = (mse > threshold).astype(int)
            
            metrics = {
                "accuracy": accuracy_score(y, pred_labels),
                "precision": precision_score(y, pred_labels, zero_division=0),
                "recall": recall_score(y, pred_labels, zero_division=0),
                "f1": f1_score(y, pred_labels, zero_division=0),
                "confusion_matrix": confusion_matrix(y, pred_labels).tolist(),
                "optimal_threshold": threshold,
                "reconstruction_stats": {
                    "min": float(np.min(mse)),
                    "max": float(np.max(mse)),
                    "mean": float(np.mean(mse)),
                    "std": float(np.std(mse))
                }
            }
        
        # Save model and reconstruction errors
        model_path = os.path.join(self.model_dir, dataset, "autoencoder.h5")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        autoencoder.save(model_path)
        
        # Add reconstruction errors to dataframe
        df["reconstruction_error"] = mse
        df["anomaly_prediction"] = (mse > config["reconstruction_threshold"]).astype(int)
        
        # Save results with reconstruction errors
        results_path = os.path.join(self.result_dir, dataset, "autoencoder_results.csv")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        df.to_csv(results_path, index=False)
        
        logger.info(f"Autoencoder metrics: {metrics}")
        return {
            "metrics": metrics,
            "model_path": model_path,
            "reconstruction_threshold": config["reconstruction_threshold"],
            "history": history.history
        }

    def _save_dataset_results(self, dataset: str, results: Dict) -> None:
        """Save evaluation results for a single dataset."""
        result_dir = os.path.join(self.result_dir, dataset)
        create_dir_if_not_exists(result_dir)
        
        # Save metrics
        metrics_path = os.path.join(result_dir, "metrics.json")
        save_json(results, metrics_path)
        
        # Save visualizations
        self._plot_unsupervised_results(dataset, results, result_dir)

    def _plot_unsupervised_results(self, dataset: str, results: Dict, output_dir: str) -> None:
        """Plot and save visualization of unsupervised results."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Plot clustering results if available
        if "kmeans" in results or "dbscan" in results:
            # Load the vectorized data
            X, y, _ = self.load_data(dataset)
            
            # Reduce dimensionality for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            plt.figure(figsize=(15, 5))
            
            # Plot true labels (if available)
            if y is not None:
                plt.subplot(1, 3, 1)
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="viridis")
                plt.title(f"True Labels - {dataset}")
            
            # Plot KMeans clusters
            if "kmeans" in results:
                with open(results["kmeans"]["model_path"], "rb") as f:
                    kmeans = pickle.load(f)
                clusters = kmeans.predict(X)
                
                plt.subplot(1, 3, 2)
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis")
                plt.title(f"KMeans Clusters - {dataset}")
            
            # Plot DBSCAN clusters
            if "dbscan" in results:
                with open(results["dbscan"]["model_path"], "rb") as f:
                    dbscan = pickle.load(f)
                clusters = dbscan.fit_predict(X)
                
                plt.subplot(1, 3, 3)
                sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="viridis")
                plt.title(f"DBSCAN Clusters - {dataset}")
            
            # Save figure
            plot_path = os.path.join(output_dir, "clustering_visualization.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved clustering visualization to {plot_path}")
        
        # Plot autoencoder results if available
        if "autoencoder" in results:
            # Load results with reconstruction errors
            results_path = os.path.join(output_dir, "autoencoder_results.csv")
            df = pd.read_csv(results_path)
            
            # Plot reconstruction error distribution
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(df["reconstruction_error"], kde=True)
            plt.axvline(results["autoencoder"]["reconstruction_threshold"], color='r')
            plt.title("Reconstruction Error Distribution")
            
            # Plot error vs threshold
            if "label" in df.columns:
                plt.subplot(1, 2, 2)
                sns.boxplot(x="label", y="reconstruction_error", data=df)
                plt.axhline(results["autoencoder"]["reconstruction_threshold"], color='r')
                plt.title("Reconstruction Error by True Label")
            
            # Save figure
            plot_path = os.path.join(output_dir, "autoencoder_analysis.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved autoencoder analysis to {plot_path}")

    def _save_combined_results(self) -> None:
        """Save combined evaluation results across all datasets."""
        combined_path = os.path.join(self.result_dir, "combined_results.json")
        save_json(self.evaluation_results, combined_path)
        
        # Create summary report
        summary = {}
        for dataset, models in self.evaluation_results.items():
            summary[dataset] = {
                model_name: {
                    "metrics": {
                        k: v for k, v in model_info["metrics"].items()
                        if k != "confusion_matrix"
                    }
                }
                for model_name, model_info in models.items()
            }
        
        summary_path = os.path.join(self.result_dir, "summary_report.json")
        save_json(summary, summary_path)
        logger.info(f"Saved combined results to {combined_path}")
        logger.info(f"Saved summary report to {summary_path}")

    def get_results(self) -> Dict:
        """Return evaluation results."""
        return self.evaluation_results