"""
Metrics Module - Evaluation metrics cho model và strategy.

Model Metrics:
- MAE, RMSE, MAPE
- Direction Accuracy

Strategy Metrics:
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


class ModelMetrics:
    """Metrics để đánh giá model performance."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Direction Accuracy - Tỷ lệ dự đoán đúng hướng.

        Returns:
            Accuracy từ 0 đến 1
        """
        # Calculate actual direction
        actual_direction = np.sign(np.diff(y_true))
        # Calculate predicted direction
        pred_direction = np.sign(y_pred[1:] - y_true[:-1])

        correct = np.sum(actual_direction == pred_direction)
        return correct / len(actual_direction)

    @staticmethod
    def evaluate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Tính tất cả model metrics.

        Returns:
            Dict với tất cả metrics
        """
        return {
            "mae": ModelMetrics.mae(y_true, y_pred),
            "rmse": ModelMetrics.rmse(y_true, y_pred),
            "mape": ModelMetrics.mape(y_true, y_pred),
            "direction_accuracy": ModelMetrics.direction_accuracy(y_true, y_pred),
        }


class StrategyMetrics:
    """Metrics để đánh giá trading strategy."""

    @staticmethod
    def total_return(initial_value: float, final_value: float) -> float:
        """Tổng lợi nhuận (%)."""
        return (final_value - initial_value) / initial_value * 100

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        """
        Sharpe Ratio.

        Args:
            returns: Daily returns
            risk_free_rate: Lãi suất phi rủi ro (annual)
            periods_per_year: Số ngày giao dịch/năm

        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)

    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> float:
        """
        Maximum Drawdown (%).

        Args:
            portfolio_values: Giá trị portfolio theo thời gian

        Returns:
            Max drawdown (số dương)
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return np.max(drawdown) * 100

    @staticmethod
    def win_rate(trades_pnl: np.ndarray) -> float:
        """
        Win Rate - Tỷ lệ trade có lãi.

        Args:
            trades_pnl: P&L của từng trade

        Returns:
            Win rate từ 0 đến 1
        """
        if len(trades_pnl) == 0:
            return 0.0
        return np.sum(trades_pnl > 0) / len(trades_pnl)

    @staticmethod
    def profit_factor(trades_pnl: np.ndarray) -> float:
        """
        Profit Factor = Gross Profit / Gross Loss.

        Args:
            trades_pnl: P&L của từng trade

        Returns:
            Profit Factor
        """
        gross_profit = np.sum(trades_pnl[trades_pnl > 0])
        gross_loss = abs(np.sum(trades_pnl[trades_pnl < 0]))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def evaluate_all(
        portfolio_values: np.ndarray,
        trades_pnl: np.ndarray,
    ) -> Dict[str, float]:
        """
        Tính tất cả strategy metrics.

        Returns:
            Dict với tất cả metrics
        """
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        return {
            "total_return": StrategyMetrics.total_return(
                portfolio_values[0], portfolio_values[-1]
            ),
            "sharpe_ratio": StrategyMetrics.sharpe_ratio(returns),
            "max_drawdown": StrategyMetrics.max_drawdown(portfolio_values),
            "win_rate": StrategyMetrics.win_rate(trades_pnl),
            "profit_factor": StrategyMetrics.profit_factor(trades_pnl),
        }


if __name__ == "__main__":
    # Test metrics
    print("=== Model Metrics Test ===")
    y_true = np.array([100, 102, 101, 105, 108])
    y_pred = np.array([101, 103, 100, 106, 107])

    metrics = ModelMetrics.evaluate_all(y_true, y_pred)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== Strategy Metrics Test ===")
    portfolio = np.array([100, 105, 103, 110, 108, 115])
    trades = np.array([5, -2, 7, -2, 7])

    strategy_metrics = StrategyMetrics.evaluate_all(portfolio, trades)
    for name, value in strategy_metrics.items():
        print(f"{name}: {value:.4f}")
