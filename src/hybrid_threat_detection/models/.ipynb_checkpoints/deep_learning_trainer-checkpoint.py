import os
import pickle
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from datetime import datetime
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, MaxPooling1D, 
    LSTM, Dense, Dropout, BatchNormalization,
    Input, Flatten, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, 
    ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.utils import to_categorical
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

class DeepLearningTrainer:
    """Production-grade deep learning model training for threat detection."""

    def __init__(self, config_path: str = None):
        setup_logging()
        self.models = {}
        self.histories = {}
        self.evaluation_results = {}

        # Create output directories
        self.model_dir = os.path.join(RESULTS_DIR, "phase3", "dl_models")
        self.result_dir = os.path.join(RESULTS_DIR, "phase3", "dl_results")
        self.tensorboard_dir = os.path.join(RESULTS_DIR, "phase3", "tensorboard_logs")
        create_dir_if_not_exists(self.model_dir)
        create_dir_if_not_exists(self.result_dir)
        create_dir_if_not_exists(self.tensorboard_dir)

        # Load configuration
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict:
        """Load deep learning configuration from JSON file."""
        default_config = {
            "models": {
                "cnn": {
                    "embedding_dim": 100,
                    "filters": [64, 128],
                    "kernel_sizes": [3, 5],
                    "pool_size": 2,
                    "dense_units": [64, 32],
                    "dropout_rate": 0.5,
                    "batch_norm": True,
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "epochs": 50
                },
                "lstm": {
                    "embedding_dim": 100,
                    "lstm_units": [64, 32],
                    "dense_units": [32],
                    "dropout_rate": 0.3,
                    "recurrent_dropout": 0.2,
                    "batch_norm": True,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 50
                }
            },
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_loss",
                    "patience": 5,
                    "restore_best_weights": True
                },
                "model_checkpoint": {
                    "monitor": "val_f1",
                    "mode": "max",
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
            logger.info(f"Loaded DL config from {config_path}")
            return config
        else:
            logger.info("Using default DL configuration")
            return default_config

    def load_data(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Load tokenized sequences and splits for a dataset."""
        try:
            # Load train/test splits
            train_df = pd.read_csv(f"data/splits/{dataset_name}/train.csv")
            test_df = pd.read_csv(f"data/splits/{dataset_name}/test.csv")

            # Load tokenizer
            tokenizer_path = f"src/hybrid_threat_detection/results/phase2/vectorizers/{dataset_name}_tokenizer.pkl"
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)

            # Load padded sequences
            sequences_path = f"src/hybrid_threat_detection/results/phase2/vectorizers/{dataset_name}_sequences.npy"
            padded_sequences = np.load(sequences_path, allow_pickle=True)

            # Get corresponding indices for train/test
            train_indices = train_df.index
            test_indices = test_df.index

            X_train = padded_sequences[train_indices]
            X_test = padded_sequences[test_indices]
            y_train = train_df["label"].values
            y_test = test_df["label"].values

            # Get vocabulary info
            vocab_info = {
                "vocab_size": len(tokenizer.word_index) + 1,
                "max_len": X_train.shape[1],
                "tokenizer_config": tokenizer.get_config()
            }

            return X_train, X_test, y_train, y_test, vocab_info

        except Exception as e:
            logger.error(f"Error loading data for {dataset_name}: {str(e)}")
            raise

    def train_models(self) -> Dict[str, Any]:
        """Train and evaluate all deep learning models on all datasets."""
        results = {}

        for dataset in self.config["datasets"]:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training DL models on {dataset} dataset")
                logger.info(f"{'='*50}")

                # Load data
                X_train, X_test, y_train, y_test, vocab_info = self.load_data(dataset)
                
                # Convert labels to categorical for DL models
                y_train_cat = to_categorical(y_train)
                y_test_cat = to_categorical(y_test)

                # Train and evaluate each model
                dataset_results = {}
                for model_name in self.config["models"].keys():
                    logger.info(f"\nTraining {model_name} on {dataset}")

                    try:
                        # Create model
                        if model_name == "cnn":
                            model = self._build_cnn_model(vocab_info)
                        elif model_name == "lstm":
                            model = self._build_lstm_model(vocab_info)
                        else:
                            raise ValueError(f"Unknown model type: {model_name}")

                        # Prepare callbacks
                        callbacks = self._prepare_callbacks(dataset, model_name)

                        # Train model
                        history = model.fit(
                            X_train, y_train_cat,
                            validation_split=0.2,
                            batch_size=self.config["models"][model_name]["batch_size"],
                            epochs=self.config["models"][model_name]["epochs"],
                            callbacks=callbacks,
                            verbose=1
                        )

                        # Evaluate on test set
                        metrics = self._evaluate_model(model, X_test, y_test)

                        # Save results
                        dataset_results[model_name] = {
                            "metrics": metrics,
                            "model_path": self._save_model(model, dataset, model_name),
                            "vocab_info": vocab_info,
                            "history": history.history
                        }

                        self.histories[(dataset, model_name)] = history

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

    def _build_cnn_model(self, vocab_info: Dict) -> Model:
        """Build a 1D CNN model for text classification."""
        config = self.config["models"]["cnn"]
        vocab_size = vocab_info["vocab_size"]
        max_len = vocab_info["max_len"]
        embedding_dim = config["embedding_dim"]

        model = Sequential()
        
        # Embedding layer
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len
        ))

        # Multiple Conv1D + MaxPooling blocks
        for filters, kernel_size in zip(config["filters"], config["kernel_sizes"]):
            model.add(Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            ))
            if config["batch_norm"]:
                model.add(BatchNormalization())
            model.add(MaxPooling1D(
                pool_size=config["pool_size"]
            ))

        model.add(Flatten())

        # Dense layers
        for units in config["dense_units"]:
            model.add(Dense(units, activation='relu'))
            if config["batch_norm"]:
                model.add(BatchNormalization())
            model.add(Dropout(config["dropout_rate"]))

        # Output layer
        model.add(Dense(2, activation='softmax'))

        # Compile model
        optimizer = Adam(learning_rate=config["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', self._f1_metric]
        )

        model.summary(print_fn=logger.info)
        return model

    def _build_lstm_model(self, vocab_info: Dict) -> Model:
        """Build an LSTM model for text classification."""
        config = self.config["models"]["lstm"]
        vocab_size = vocab_info["vocab_size"]
        max_len = vocab_info["max_len"]
        embedding_dim = config["embedding_dim"]

        model = Sequential()

        # Embedding layer
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len
        ))

        # Stacked LSTM layers
        for i, units in enumerate(config["lstm_units"]):
            return_sequences = i < len(config["lstm_units"]) - 1
            model.add(LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=config["dropout_rate"],
                recurrent_dropout=config["recurrent_dropout"]
            ))
            if config["batch_norm"]:
                model.add(BatchNormalization())

        # Dense layers
        for units in config["dense_units"]:
            model.add(Dense(units, activation='relu'))
            if config["batch_norm"]:
                model.add(BatchNormalization())
            model.add(Dropout(config["dropout_rate"]))

        # Output layer
        model.add(Dense(2, activation='softmax'))

        # Compile model
        optimizer = Adam(learning_rate=config["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', self._f1_metric]
        )

        model.summary(print_fn=logger.info)
        return model

    def _f1_metric(self, y_true, y_pred):
        """Custom metric function for F1 score."""
        from tensorflow.keras import backend as K
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    def _prepare_callbacks(self, dataset: str, model_name: str) -> List:
        """Prepare training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(**self.config["callbacks"]["early_stopping"])
        callbacks.append(early_stop)
        
        # Model checkpoint
        model_path = os.path.join(self.model_dir, dataset, f"{model_name}_best.h5")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            **self.config["callbacks"]["model_checkpoint"]
        )
        callbacks.append(checkpoint)
        
        # Reduce LR on plateau
        reduce_lr = ReduceLROnPlateau(**self.config["callbacks"]["reduce_lr"])
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tb_log_dir = os.path.join(self.tensorboard_dir, dataset, model_name)
        os.makedirs(tb_log_dir, exist_ok=True)
        tensorboard = TensorBoard(log_dir=tb_log_dir)
        callbacks.append(tensorboard)
        
        return callbacks

    def _evaluate_model(self, model: Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance on test set."""
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_prob = y_pred_probs[:, 1]  # Probability of positive class

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        logger.info(f"Evaluation metrics:\n{pd.DataFrame([metrics])}")
        return metrics

    def _save_model(self, model: Model, dataset: str, model_name: str) -> str:
        """Save trained model to disk."""
        model_dir = os.path.join(self.model_dir, dataset)
        create_dir_if_not_exists(model_dir)
        
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(model_path)
        
        logger.info(f"Saved {model_name} model to {model_path}")
        return model_path

    def _save_dataset_results(self, dataset: str, results: Dict) -> None:
        """Save evaluation results for a single dataset."""
        result_dir = os.path.join(self.result_dir, dataset)
        create_dir_if_not_exists(result_dir)
        
        # Save metrics
        metrics_path = os.path.join(result_dir, "metrics.json")
        save_json(results, metrics_path)
        
        # Save training history plots
        self._plot_training_history(dataset, results, result_dir)
        
        # Save confusion matrices
        self._plot_confusion_matrices(dataset, results, result_dir)

    def _plot_training_history(self, dataset: str, results: Dict, output_dir: str) -> None:
        """Plot and save training history metrics."""
        import matplotlib.pyplot as plt
        
        for model_name, result in results.items():
            history = result["history"]
            
            # Plot accuracy
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'], label='Train Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{model_name} Accuracy ({dataset})')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title(f'{model_name} Loss ({dataset})')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            # Save figure
            plot_path = os.path.join(output_dir, f"{model_name}_training_history.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved training history to {plot_path}")

    def _plot_confusion_matrices(self, dataset: str, results: Dict, output_dir: str) -> None:
        """Plot and save confusion matrices."""
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