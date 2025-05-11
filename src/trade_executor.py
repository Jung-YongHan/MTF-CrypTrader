from typing import Any, Dict

from src.portfoilo_manager import PortfolioManager


class TradeExecutor:

    async def execute(
        self,
        price_data: Dict[str, Any],
        coin: str,
        micro_report: Dict[str, Any] = None,
    ) -> None:
        if micro_report is None:
            return

        order_report = micro_report["order_report"]

        order_type = order_report["order"]
        amount = order_report["amount"]  # 주문 비율

        await PortfolioManager.get_instance().update_portfolio_by_trade(
            price_data=price_data,
            coin=coin,
            amount=amount,
            order_type=order_type,
        )

        print(
            f"Trade executed: {order_type} {amount} of {coin} at price {price_data.get("open")}."
        )
