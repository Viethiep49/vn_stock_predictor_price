"""
Model Trainer Module - Training logic cho LSTM model.

Chức năng:
- Training với callbacks (EarlyStopping, ReduceLROnPlateau)
- Model checkpointing
- Training history logging
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from .lstm_attention import LSTMAttentionModel


class ModelTrainer:
    """Trainer cho LSTM + Attention model."""

    def __init__(
        self,
        model: LSTMAttentionModel,
        model_dir: str = "data/models",
        log_dir: str = "logs",
    ):
        """
        Khởi tạo trainer.

        Args:
            model: LSTMAttentionModel instance
            model_dir: Thư mục lưu model
            log_dir: Thư mục lưu logs
        """
        self.model = model.get_model()
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = None

    def _get_callbacks(
        self,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 5,
        model_name: str = "mbb_lstm_attention",
    ) -> list:
        """
        Tạo callbacks cho training.

        Args:
            early_stopping_patience: Patience cho early stopping
            reduce_lr_patience: Patience cho reduce learning rate
            model_name: Tên model để save

        Returns:
            List các callbacks
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
            # Model checkpoint
            ModelCheckpoint(
                filepath=str(self.model_dir / f"{model_name}_best.keras"),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            # TensorBoard
            TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
            ),
        ]

        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 5,
    ) -> Dict:
        """
        Train model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Số epochs
            batch_size: Batch size
            early_stopping_patience: Patience cho early stopping
            reduce_lr_patience: Patience cho reduce LR

        Returns:
            Training history dict
        """
        callbacks = self._get_callbacks(
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience,
        )

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Evaluate model trên test set.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            (loss, mae, mse)
        """
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return results  # [loss, mae, mse]

    def save_model(self, filename: str = "mbb_lstm_attention_final"):
        """
        Lưu model.

        Args:
            filename: Tên file (không cần extension)
        """
        filepath = self.model_dir / f"{filename}.keras"
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename: str = "mbb_lstm_attention_final"):
        """
        Load model từ file.

        Args:
            filename: Tên file (không cần extension)
        """
        from .lstm_attention import AttentionLayer

        filepath = self.model_dir / f"{filename}.keras"
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={"AttentionLayer": AttentionLayer},
        )
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test trainer setup
    model = LSTMAttentionModel()
    trainer = ModelTrainer(model)
    print(f"Model dir: {trainer.model_dir}")
    print(f"Log dir: {trainer.log_dir}")
