"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np

from src.features.technical import TechnicalFeatures


class TestTechnicalFeatures:
    """Test TechnicalFeatures class."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n_days = 100

        dates = pd.date_range(end="2024-01-01", periods=n_days, freq="D")
        close = 25000 + np.cumsum(np.random.randn(n_days) * 100)

        df = pd.DataFrame(
            {
                "open": close - np.random.rand(n_days) * 100,
                "high": close + np.random.rand(n_days) * 200,
                "low": close - np.random.rand(n_days) * 200,
                "close": close,
                "volume": np.random.randint(1000000, 5000000, n_days),
            },
            index=dates,
        )
        return df

    def test_add_base_features(self, sample_ohlcv):
        """Test adding base features."""
        tf = TechnicalFeatures()
        df = tf.add_base_features(sample_ohlcv)

        # Check if all base features are added
        expected_columns = [
            "sma_5", "sma_20", "sma_50",
            "ema_12", "ema_26",
            "rsi_14",
            "macd", "macd_signal",
            "bb_upper", "bb_lower",
            "atr_14",
            "price_change", "volume_change",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_add_advanced_technical(self, sample_ohlcv):
        """Test adding advanced technical features."""
        tf = TechnicalFeatures()
        df = tf.add_advanced_technical(sample_ohlcv)

        expected_columns = ["adx", "obv", "stoch_k", "stoch_d"]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_feature_columns_count(self):
        """Test that feature columns list has correct count."""
        columns = TechnicalFeatures.get_feature_columns()
        # We expect 24 features (excluding market and flow which need external data)
        assert len(columns) == 24, f"Expected 24 columns, got {len(columns)}"

    def test_calculate_all_features(self, sample_ohlcv):
        """Test calculating all features."""
        tf = TechnicalFeatures()
        df = tf.calculate_all_features(sample_ohlcv)

        # Should have no NaN after dropna
        assert df.isna().sum().sum() == 0, "DataFrame should have no NaN values"

        # Should have all feature columns
        expected_count = len(TechnicalFeatures.get_feature_columns())
        actual_count = len([c for c in df.columns if c in TechnicalFeatures.get_feature_columns()])
        assert actual_count == expected_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
