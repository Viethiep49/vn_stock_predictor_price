"""
Data Collector Module - Thu thập dữ liệu từ Vnstock API.

Chức năng:
- Lấy dữ liệu OHLCV cho MBB (2015-2024)
- Lấy dữ liệu VN-Index, VN30 cho market context
- Lấy dữ liệu foreign trading flow
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from vnstock import Vnstock


class DataCollector:
    """Thu thập dữ liệu chứng khoán từ Vnstock."""

    def __init__(self, symbol: str = "MBB"):
        """
        Khởi tạo DataCollector.

        Args:
            symbol: Mã cổ phiếu (default: MBB)
        """
        self.symbol = symbol
        self.stock = Vnstock().stock(symbol=symbol, source="VCI")

    def get_ohlcv(
        self,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        interval: str = "1D",
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu OHLCV.

        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (default: hôm nay)
            interval: Khung thời gian ("1D" = daily, "1W" = weekly)

        Returns:
            DataFrame với columns: open, high, low, close, volume
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        df = self.stock.quote.history(
            start=start_date,
            end=end_date,
            interval=interval,
        )
        return df

    def get_market_index(
        self,
        index_symbol: str = "VNINDEX",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu chỉ số thị trường.

        Args:
            index_symbol: Mã chỉ số (VNINDEX, VN30, HNX)
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc

        Returns:
            DataFrame với dữ liệu index
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        index_stock = Vnstock().stock(symbol=index_symbol, source="VCI")
        df = index_stock.quote.history(
            start=start_date,
            end=end_date,
            interval="1D",
        )
        return df

    def get_foreign_trading(
        self,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Lấy dữ liệu giao dịch khối ngoại.

        Args:
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc

        Returns:
            DataFrame với foreign buy/sell volume
        """
        # TODO: Implement foreign trading data collection
        # Vnstock có thể cần API khác cho dữ liệu này
        raise NotImplementedError("Foreign trading data not yet implemented")

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Lưu dữ liệu ra file CSV.

        Args:
            df: DataFrame cần lưu
            filename: Tên file (không cần extension)
        """
        filepath = f"data/raw/{filename}.csv"
        df.to_csv(filepath, index=True)
        print(f"Data saved to {filepath}")

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Đọc dữ liệu từ file CSV.

        Args:
            filename: Tên file (không cần extension)

        Returns:
            DataFrame
        """
        filepath = f"data/raw/{filename}.csv"
        return pd.read_csv(filepath, index_col=0, parse_dates=True)


if __name__ == "__main__":
    # Data collection and save
    collector = DataCollector("MBB")

    print("Fetching MBB daily data...")
    df_daily = collector.get_ohlcv(start_date="2015-01-01", interval="1D")
    print(f"Daily data shape: {df_daily.shape}")
    print(df_daily.head())
    collector.save_data(df_daily, "mbb_daily")

    print("\nFetching MBB weekly data...")
    df_weekly = collector.get_ohlcv(start_date="2015-01-01", interval="1W")
    print(f"Weekly data shape: {df_weekly.shape}")
    print(df_weekly.head())
    collector.save_data(df_weekly, "mbb_weekly")

    print("\nFetching VNINDEX data...")
    df_vnindex = collector.get_market_index(index_symbol="VNINDEX", start_date="2015-01-01")
    print(f"VNINDEX data shape: {df_vnindex.shape}")
    collector.save_data(df_vnindex, "vnindex_daily")

    print("\nFetching VN30 data...")
    df_vn30 = collector.get_market_index(index_symbol="VN30", start_date="2015-01-01")
    print(f"VN30 data shape: {df_vn30.shape}")
    collector.save_data(df_vn30, "vn30_daily")

    print("\n=== Data collection completed! ===")
