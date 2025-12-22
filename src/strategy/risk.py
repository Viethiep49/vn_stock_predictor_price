"""
Risk Management Module - Quản lý rủi ro giao dịch.

Chiến lược:
- Position Size: Fixed 5% portfolio
- Stop Loss: Fixed -5%
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    """Position data class."""

    symbol: str
    entry_price: float
    quantity: int
    entry_date: str
    stop_loss_price: float
    current_price: Optional[float] = None


class RiskManager:
    """Quản lý rủi ro cho trading."""

    def __init__(
        self,
        portfolio_value: float,
        position_size_pct: float = 0.05,
        stop_loss_pct: float = 0.05,
        max_holding_days: int = 20,
    ):
        """
        Khởi tạo RiskManager.

        Args:
            portfolio_value: Giá trị portfolio
            position_size_pct: % portfolio cho mỗi trade (default: 5%)
            stop_loss_pct: % stop loss (default: 5%)
            max_holding_days: Số ngày giữ tối đa
        """
        self.portfolio_value = portfolio_value
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days

        self.current_position: Optional[Position] = None

    def calculate_position_size(self, price: float) -> int:
        """
        Tính số lượng cổ phiếu để mua.

        Args:
            price: Giá cổ phiếu

        Returns:
            Số lượng cổ phiếu (làm tròn 100)
        """
        max_investment = self.portfolio_value * self.position_size_pct
        quantity = int(max_investment / price)

        # Round down to 100 (lô 100 cổ phiếu)
        quantity = (quantity // 100) * 100

        return max(quantity, 0)

    def calculate_stop_loss(self, entry_price: float) -> float:
        """
        Tính giá stop loss.

        Args:
            entry_price: Giá mua

        Returns:
            Giá stop loss
        """
        return entry_price * (1 - self.stop_loss_pct)

    def should_stop_loss(self, current_price: float) -> bool:
        """
        Kiểm tra có nên stop loss không.

        Args:
            current_price: Giá hiện tại

        Returns:
            True nếu cần stop loss
        """
        if self.current_position is None:
            return False

        return current_price <= self.current_position.stop_loss_price

    def open_position(
        self,
        symbol: str,
        price: float,
        date: str,
    ) -> Optional[Position]:
        """
        Mở position mới.

        Args:
            symbol: Mã cổ phiếu
            price: Giá mua
            date: Ngày mua

        Returns:
            Position mới hoặc None nếu đã có position
        """
        if self.current_position is not None:
            print("Already have an open position.")
            return None

        quantity = self.calculate_position_size(price)
        if quantity == 0:
            print("Position size too small.")
            return None

        stop_loss = self.calculate_stop_loss(price)

        self.current_position = Position(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            entry_date=date,
            stop_loss_price=stop_loss,
            current_price=price,
        )

        return self.current_position

    def close_position(self, price: float) -> Optional[float]:
        """
        Đóng position.

        Args:
            price: Giá bán

        Returns:
            P&L hoặc None nếu không có position
        """
        if self.current_position is None:
            return None

        pnl = (price - self.current_position.entry_price) * self.current_position.quantity
        pnl_pct = (price - self.current_position.entry_price) / self.current_position.entry_price

        print(f"Position closed:")
        print(f"  Entry: {self.current_position.entry_price}")
        print(f"  Exit: {price}")
        print(f"  Quantity: {self.current_position.quantity}")
        print(f"  P&L: {pnl:,.0f} VND ({pnl_pct:.2%})")

        self.current_position = None
        return pnl

    def update_portfolio_value(self, new_value: float):
        """
        Cập nhật giá trị portfolio.

        Args:
            new_value: Giá trị portfolio mới
        """
        self.portfolio_value = new_value


if __name__ == "__main__":
    # Test risk manager
    rm = RiskManager(portfolio_value=100_000_000)  # 100M VND

    print(f"Portfolio: {rm.portfolio_value:,.0f} VND")
    print(f"Position size: {rm.position_size_pct:.0%}")
    print(f"Stop loss: {rm.stop_loss_pct:.0%}")

    # Test position sizing
    price = 25000
    qty = rm.calculate_position_size(price)
    print(f"\nPrice: {price:,.0f} VND")
    print(f"Quantity: {qty} shares")
    print(f"Investment: {price * qty:,.0f} VND")
