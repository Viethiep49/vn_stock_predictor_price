"""
Stock Predictor Module - Inference và prediction.

Chức năng:
- Load trained model
- Predict giá ngày tiếp theo
- Generate trading signals
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


class StockPredictor:
    """Predictor cho stock price prediction."""

    def __init__(
        self,
        model_path: str = "data/models/mbb_lstm_attention_best.keras",
        signal_threshold: float = 0.05,
    ):
        """
        Khởi tạo predictor.

        Args:
            model_path: Đường dẫn đến trained model
            signal_threshold: Ngưỡng để generate signals (default: 5%)
        """
        self.model_path = Path(model_path)
        self.signal_threshold = signal_threshold
        self.model = None

        if self.model_path.exists():
            self.load_model()

    def load_model(self):
        """Load model từ file."""
        from .lstm_attention import AttentionLayer

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"AttentionLayer": AttentionLayer},
        )
        print(f"Model loaded from {self.model_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict giá.

        Args:
            X: Input features (batch, sequence_length, num_features)

        Returns:
            Predicted prices (scaled)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        return self.model.predict(X, verbose=0)

    def predict_next_day(
        self,
        X: np.ndarray,
        current_price: float,
        price_scaler,
    ) -> Tuple[float, str, float]:
        """
        Predict giá ngày tiếp theo và generate signal.

        Args:
            X: Input features cho 1 sample (1, 60, 28)
            current_price: Giá hiện tại
            price_scaler: Scaler đã fit (để inverse transform)

        Returns:
            (predicted_price, signal, predicted_change_pct)
        """
        # Predict
        pred_scaled = self.predict(X)

        # Inverse transform
        pred_price = price_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

        # Calculate change
        change_pct = (pred_price - current_price) / current_price

        # Generate signal
        signal = self.generate_signal(change_pct)

        return pred_price, signal, change_pct

    def generate_signal(self, predicted_change: float) -> str:
        """
        Generate trading signal.

        Args:
            predicted_change: Dự đoán % thay đổi giá

        Returns:
            "BUY", "SELL", hoặc "HOLD"
        """
        if predicted_change >= self.signal_threshold:
            return "BUY"
        elif predicted_change <= -self.signal_threshold:
            return "SELL"
        else:
            return "HOLD"

    def get_prediction_confidence(
        self,
        X: np.ndarray,
        n_iterations: int = 100,
    ) -> Tuple[float, float]:
        """
        Tính confidence qua Monte Carlo Dropout.

        Args:
            X: Input features
            n_iterations: Số lần predict với dropout

        Returns:
            (mean_prediction, std_prediction)
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Enable training mode for dropout
        predictions = []
        for _ in range(n_iterations):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        return mean_pred, std_pred


if __name__ == "__main__":
    # Test predictor
    predictor = StockPredictor()
    print(f"Signal threshold: {predictor.signal_threshold}")
    print(f"Model path: {predictor.model_path}")
