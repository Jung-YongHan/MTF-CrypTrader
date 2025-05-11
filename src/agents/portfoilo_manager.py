from typing import Any, Dict


class PortfolioManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    # get_instance 메서드 추가
    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        coin: str,
        cash: float,
    ):
        self.portfolio = {
            "cash": cash,
            coin: 0,
        }
        self.portfolio_ratio = {
            "cash": 1,
            coin: 0,
        }
        self.coin = coin
        self.fee = 0.0008

    async def update_portfolio_ratio(self, price: float) -> None:
        """_summary_
        매 캔들마다 포트폴리오 비율을 업데이트합니다.
        현재 현금과 코인의 비율을 캔들 시가 기준으로 업데이트합니다.

        Args:
            price_data (Dict[str, Any]): _description_

        Raises:
            ValueError: _description_
        """

        cash_amount = self.portfolio.get("cash")  # 현금 보유량
        coin_amount = self.portfolio.get(self.coin)  # 코인 보유량

        total_amount = cash_amount + coin_amount * price

        # Update portfolio ratio
        self.portfolio_ratio["cash"] = cash_amount / total_amount
        self.portfolio_ratio[self.coin] = coin_amount * price / total_amount

    def update_portfolio_by_trade(
        self,
        price: float,
        coin: str,
        amount: float,
        order_type: str,
    ) -> None:
        """_summary_
        매 거래마다 포트폴리오를 업데이트합니다.
        어떤 자산(현금, 코인)에 대해서 업데이트할 것인지 결정합니다.

        Args:
            price (float): 시가
            coin (str): 코인 종류
            amount (float): 주문할 코인의 총 자산 대비 비율
            order_type (str): "long", "short" or "hold"

        Raises:
            ValueError: _description_
        """
        if coin not in self.portfolio:
            raise ValueError(f"Coin {coin} not in portfolio.")

        total_amount = self.portfolio["cash"] + self.portfolio[coin] * price  # 총 자산

        if order_type == "long":
            # 수수료가 반영된 매수할 원화 금액
            total_amount_after_fee = total_amount * amount * (1 - self.fee)
            # 구매할 코인 수량
            pay_amount = total_amount_after_fee / price
            # 최종 코인 수량
            self.portfolio[self.coin] += pay_amount
            # 최종 현금 수량, 수수료가 반영되지 않은 현금에서 차감
            self.portfolio["cash"] -= total_amount * amount

        elif order_type == "short":
            # 매도할 원화 금액
            sell_value = total_amount * amount
            # 판매할 코인 수량
            pay_amount = sell_value / price
            # 최종 코인 수량
            self.portfolio[self.coin] -= pay_amount
            # 최종 현금 수량, 수수료가 반영된 현금에서 차감
            self.portfolio["cash"] += sell_value * (1 - self.fee)

        self.update_portfolio_ratio(price)

    # 모든 코인을 주어진 코인 가격에 맞게 현금으로 판매
    def sell_all(self, price: float) -> None:
        """_summary_
        모든 코인을 주어진 코인 가격에 맞게 현금으로 판매합니다.

        Args:
            price (float): 시가

        Raises:
            ValueError: _description_
        """
        if self.coin not in self.portfolio:
            raise ValueError(f"Coin {self.coin} not in portfolio.")

        coin_amount = self.portfolio[self.coin]
        self.portfolio["cash"] += (coin_amount * price) * (1 - self.fee)
        self.portfolio[self.coin] = 0
        self.update_portfolio_ratio(price)

    def get_portfolio(self) -> Dict[str, Any]:
        return self.portfolio

    def get_portfolio_ratio(self) -> Dict[str, float]:
        return self.portfolio_ratio
