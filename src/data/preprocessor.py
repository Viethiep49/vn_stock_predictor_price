"""
Data Preprocessor Module - Xử lý và chuẩn bị dữ liệu cho model.

Chức năng:
- Làm sạch dữ liệu (handle missing values, outliers)
- Tạo sequences cho LSTM
- Normalize/scale dữ liệu
- Train/Val/Test split
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """Xử lý và chuẩn bị dữ liệu cho training."""

    def __init__(
        self,
        sequence_length: int = 60,
        scaler_type: str = "minmax",
    ):
        """
        Khởi tạo DataPreprocessor.

        Args:
            sequence_length: Số ngày lookback cho LSTM
            scaler_type: Loại scaler ("minmax" hoặc "standard")
        """
        self.sequence_length = sequence_length
        self.scaler_type = scaler_type

        if scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = StandardScaler()

        self.price_scaler = MinMaxScaler(feature_range=(0, 1))

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu.

        Args:
            df: DataFrame gốc

        Returns:
            DataFrame đã làm sạch
        """
        df = df.copy()

        # Drop rows với missing values
        df = df.dropna()

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Sort by date
        df = df.sort_index()

        return df

    def create_sequences(
        self,
        features: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo sequences cho LSTM.

        Args:
            features: Feature array (n_samples, n_features)
            target: Target array (n_samples,)

        Returns:
            X: (n_sequences, sequence_length, n_features)
            y: (n_sequences,)
        """
        X, y = [], []

        for i in range(self.sequence_length, len(features)):
            X.append(features[i - self.sequence_length : i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, ...]:
        """
        Chia dữ liệu thành train/val/test (time-based split).

        Args:
            X: Feature sequences
            y: Target values
            train_ratio: Tỷ lệ training data
            val_ratio: Tỷ lệ validation data

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size : train_size + val_size]
        y_val = y[train_size : train_size + val_size]

        X_test = X[train_size + val_size :]
        y_test = y[train_size + val_size :]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler và transform data.

        Args:
            df: DataFrame với features

        Returns:
            Scaled numpy array
        """
        return self.scaler.fit_transform(df.values)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data với fitted scaler.

        Args:
            df: DataFrame với features

        Returns:
            Scaled numpy array
        """
        return self.scaler.transform(df.values)

    def inverse_transform_price(self, scaled_price: np.ndarray) -> np.ndarray:
        """
        Inverse transform giá về giá trị gốc.

        Args:
            scaled_price: Giá đã scale

        Returns:
            Giá gốc
        """
        return self.price_scaler.inverse_transform(
            scaled_price.reshape(-1, 1)
        ).flatten()

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        target_column: str = "close",
    ) -> Tuple[np.ndarray, ...]:
        """
        Pipeline hoàn chỉnh để chuẩn bị dữ liệu.

        Args:
            df: DataFrame với OHLCV và features
            feature_columns: List các cột feature
            target_column: Cột target (default: close)

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Clean data
        df = self.clean_data(df)

        # Scale features
        features_scaled = self.fit_transform(df[feature_columns])

        # Scale target separately (để inverse transform sau)
        target = df[target_column].values.reshape(-1, 1)
        target_scaled = self.price_scaler.fit_transform(target).flatten()

        # Create sequences
        X, y = self.create_sequences(features_scaled, target_scaled)

        # Split data
        return self.split_data(X, y)


if __name__ == "__main__":
    # Test preprocessor
    preprocessor = DataPreprocessor(sequence_length=60)
    print(f"Sequence length: {preprocessor.sequence_length}")
    print(f"Scaler type: {preprocessor.scaler_type}")
