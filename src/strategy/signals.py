"""
Signal Generator Module - Tạo tín hiệu giao dịch.

Chiến lược:
- Threshold: ±5% (conservative)
- Signal types: BUY, SELL, HOLD
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd


@dataclass
class TradingSignal:
    """Trading signal data class."""

    date: datetime
    signal: str  # BUY, SELL, HOLD
    current_price: float
    predicted_price: float
    predicted_change: float
    confidence: Optional[float] = None


class SignalGenerator:
    """Generate trading signals từ predictions."""

    def __init__(
        self,
        buy_threshold: float = 0.05,
        sell_threshold: float = -0.05,
    ):
        """
        Khởi tạo SignalGenerator.

        Args:
            buy_threshold: Ngưỡng để BUY (default: +5%)
            sell_threshold: Ngưỡng để SELL (default: -5%)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.signals_history: List[TradingSignal] = []

    def generate_signal(
        self,
        current_price: float,
        predicted_price: float,
        date: datetime = None,
        confidence: float = None,
    ) -> TradingSignal:
        """
        Generate trading signal.

        Args:
            current_price: Giá hiện tại
            predicted_price: Giá dự đoán
            date: Ngày (default: now)
            confidence: Model confidence (optional)

        Returns:
            TradingSignal
        """
        if date is None:
            date = datetime.now()

        predicted_change = (predicted_price - current_price) / current_price

        if predicted_change >= self.buy_threshold:
            signal_type = "BUY"
        elif predicted_change <= self.sell_threshold:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"

        signal = TradingSignal(
            date=date,
            signal=signal_type,
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_change=predicted_change,
            confidence=confidence,
        )

        self.signals_history.append(signal)
        return signal

    def get_signals_dataframe(self) -> pd.DataFrame:
        """
        Convert signals history thành DataFrame.

        Returns:
            DataFrame với signals history
        """
        if not self.signals_history:
            return pd.DataFrame()

        data = [
            {
                "date": s.date,
                "signal": s.signal,
                "current_price": s.current_price,
                "predicted_price": s.predicted_price,
                "predicted_change": s.predicted_change,
                "confidence": s.confidence,
            }
            for s in self.signals_history
        ]

        return pd.DataFrame(data)

    def clear_history(self):
        """Clear signals history."""
        self.signals_history = []


if __name__ == "__main__":
    # Test signal generator
    generator = SignalGenerator()

    # Test signals
    test_cases = [
        (25000, 26500),  # +6% -> BUY
        (25000, 23500),  # -6% -> SELL
        (25000, 25500),  # +2% -> HOLD
    ]

    for current, predicted in test_cases:
        signal = generator.generate_signal(current, predicted)
        print(f"Current: {current}, Predicted: {predicted}")
        print(f"  Signal: {signal.signal}, Change: {signal.predicted_change:.2%}")
