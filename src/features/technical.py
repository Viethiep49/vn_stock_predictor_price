"""
Technical Features Module - Tính toán các chỉ số kỹ thuật.

Features (28 total):
- Base (15): OHLCV, MAs, RSI, MACD, BB, ATR
- Technical+ (5): ADX, OBV, Stochastic
- Market (4): VN-Index, VN30, USD/VND returns
- Flow (2): Foreign net, Proprietary net
"""

import pandas as pd
import pandas_ta as ta


class TechnicalFeatures:
    """Tính toán các chỉ số kỹ thuật cho stock data."""

    def __init__(self):
        """Khởi tạo TechnicalFeatures."""
        pass

    def add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm base features (15 features).

        Args:
            df: DataFrame với OHLCV data

        Returns:
            DataFrame với thêm các features
        """
        df = df.copy()

        # Moving Averages
        df["sma_5"] = ta.sma(df["close"], length=5)
        df["sma_20"] = ta.sma(df["close"], length=20)
        df["sma_50"] = ta.sma(df["close"], length=50)
        df["ema_12"] = ta.ema(df["close"], length=12)
        df["ema_26"] = ta.ema(df["close"], length=26)

        # RSI
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]

        # Bollinger Bands
        bbands = ta.bbands(df["close"], length=20, std=2)
        df["bb_upper"] = bbands["BBU_20_2.0"]
        df["bb_lower"] = bbands["BBL_20_2.0"]

        # ATR
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        # Price & Volume Changes
        df["price_change"] = df["close"].pct_change()
        df["volume_change"] = df["volume"].pct_change()

        return df

    def add_advanced_technical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm advanced technical features (Group A - 5 features).

        Args:
            df: DataFrame với OHLCV data

        Returns:
            DataFrame với thêm các features
        """
        df = df.copy()

        # ADX - Trend Strength
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx["ADX_14"]

        # OBV - On Balance Volume
        df["obv"] = ta.obv(df["close"], df["volume"])

        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

        return df

    def add_market_context(
        self,
        df: pd.DataFrame,
        vnindex_df: pd.DataFrame,
        vn30_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Thêm market context features (Group B - 4 features).

        Args:
            df: DataFrame chính
            vnindex_df: DataFrame VN-Index
            vn30_df: DataFrame VN30

        Returns:
            DataFrame với thêm market features
        """
        df = df.copy()

        # VN-Index return
        vnindex_return = vnindex_df["close"].pct_change()
        df["vnindex_return"] = df.index.map(
            lambda x: vnindex_return.get(x, 0) if x in vnindex_return.index else 0
        )

        # VN30 return
        vn30_return = vn30_df["close"].pct_change()
        df["vn30_return"] = df.index.map(
            lambda x: vn30_return.get(x, 0) if x in vn30_return.index else 0
        )

        # USD/VND change - placeholder (cần thêm data source)
        df["usdvnd_change"] = 0.0

        return df

    def add_flow_features(
        self,
        df: pd.DataFrame,
        foreign_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Thêm trading flow features (Group C - 2 features).

        Args:
            df: DataFrame chính
            foreign_df: DataFrame foreign trading (optional)

        Returns:
            DataFrame với thêm flow features
        """
        df = df.copy()

        # Placeholder - cần data từ API
        df["foreign_net"] = 0.0
        df["proprietary_net"] = 0.0

        return df

    def calculate_all_features(
        self,
        df: pd.DataFrame,
        vnindex_df: pd.DataFrame = None,
        vn30_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Tính toán tất cả features.

        Args:
            df: DataFrame với OHLCV data
            vnindex_df: DataFrame VN-Index (optional)
            vn30_df: DataFrame VN30 (optional)

        Returns:
            DataFrame với tất cả features
        """
        # Base features
        df = self.add_base_features(df)

        # Advanced technical
        df = self.add_advanced_technical(df)

        # Market context (nếu có data)
        if vnindex_df is not None and vn30_df is not None:
            df = self.add_market_context(df, vnindex_df, vn30_df)
        else:
            # Placeholder columns
            df["vnindex_return"] = 0.0
            df["vn30_return"] = 0.0
            df["usdvnd_change"] = 0.0

        # Flow features
        df = self.add_flow_features(df)

        # Drop NaN rows (do MA calculations)
        df = df.dropna()

        return df

    @staticmethod
    def get_feature_columns() -> list:
        """
        Lấy danh sách tất cả feature columns.

        Returns:
            List các tên cột feature
        """
        return [
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


if __name__ == "__main__":
    # Test features
    tf = TechnicalFeatures()
    print("Feature columns:")
    for i, col in enumerate(tf.get_feature_columns(), 1):
        print(f"  {i}. {col}")
    print(f"\nTotal features: {len(tf.get_feature_columns())}")
