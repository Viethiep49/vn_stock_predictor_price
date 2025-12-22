"""
Configuration Module - Cấu hình cho project.

Tất cả hyperparameters và settings được định nghĩa ở đây.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Cấu hình cho data collection và processing."""

    symbol: str = "MBB"
    start_date: str = "2015-01-01"
    sequence_length: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    """Cấu hình cho LSTM + Attention model."""

    # Architecture
    num_features: int = 28
    lstm_units_1: int = 128
    lstm_units_2: int = 64
    attention_units: int = 64
    dropout_rate: float = 0.2
    dense_units: int = 32

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5


@dataclass
class StrategyConfig:
    """Cấu hình cho trading strategy."""

    # Signals
    buy_threshold: float = 0.05
    sell_threshold: float = -0.05

    # Risk Management
    position_size_pct: float = 0.05
    stop_loss_pct: float = 0.05
    max_holding_days: int = 20


@dataclass
class PathConfig:
    """Cấu hình đường dẫn."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.project_root / "data" / "models"

    @property
    def logs(self) -> Path:
        return self.project_root / "logs"


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Feature columns
    feature_columns: List[str] = field(
        default_factory=lambda: [
            # Base OHLCV
            "close",
            "volume",
            # Moving Averages
            "sma_5",
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            # Momentum
            "rsi_14",
            "macd",
            "macd_signal",
            # Volatility
            "bb_upper",
            "bb_lower",
            "atr_14",
            # Changes
            "price_change",
            "volume_change",
            # Advanced Technical
            "adx",
            "obv",
            "stoch_k",
            "stoch_d",
            # Market Context
            "vnindex_return",
            "vn30_return",
            "usdvnd_change",
            # Flow
            "foreign_net",
            "proprietary_net",
        ]
    )


# Global config instance
config = Config()


if __name__ == "__main__":
    # Print configuration
    print("=== Data Config ===")
    print(f"Symbol: {config.data.symbol}")
    print(f"Start date: {config.data.start_date}")
    print(f"Sequence length: {config.data.sequence_length}")

    print("\n=== Model Config ===")
    print(f"LSTM units: {config.model.lstm_units_1}, {config.model.lstm_units_2}")
    print(f"Dropout: {config.model.dropout_rate}")
    print(f"Learning rate: {config.model.learning_rate}")

    print("\n=== Strategy Config ===")
    print(f"Buy threshold: {config.strategy.buy_threshold:.0%}")
    print(f"Stop loss: {config.strategy.stop_loss_pct:.0%}")

    print("\n=== Paths ===")
    print(f"Data raw: {config.paths.data_raw}")
    print(f"Models: {config.paths.models}")

    print(f"\n=== Features ({len(config.feature_columns)}) ===")
    for col in config.feature_columns:
        print(f"  - {col}")
