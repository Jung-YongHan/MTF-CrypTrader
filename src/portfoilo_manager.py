from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.enum.coin_type import CoinType
from src.enum.order_type import OrderType


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
        risk_free_rate: float = 0.0,
        interval_minutes: int = 15,
    ):
        self.coin = coin
        self.fee = 0.0008
        self.portfolio = {
            "cash": cash,
            coin: 0,
        }
        self.portfolio_ratio = {
            "cash": 1,
            coin: 0,
        }

        # 성과 지표 기록
        self.initial_value = cash
        self.portfolio_value_history: List[Dict[str, Any]] = [
            {"date": None, "value": cash}
        ]
        self.peak_value = cash
        self.max_drawdown = 0.0

        # 거래 기록
        self.trade_history: List[Dict[str, Any]] = []
        self._open_trade: Optional[Dict[str, Any]] = None

        # 샤프 계산용
        self.risk_free_rate = risk_free_rate
        # 연환산을 위한 인터벌 길이(분 단위), 15분봉이라면 15를 입력
        self.interval_minutes = interval_minutes

    async def update_portfolio_ratio(
        self, price_data: Dict[str, Any], is_sell_all: bool = False
    ) -> None:
        """_summary_
        매 캔들마다 포트폴리오 비율을 업데이트합니다.
        현재 현금과 코인의 비율을 캔들 시가 기준으로 업데이트합니다.

        Args:
            price_data (Dict[str, Any]): _description_

        Raises:
            ValueError: _description_
        """

        date = price_data.get("datetime")
        if is_sell_all:
            price = price_data.get("close")
        else:
            price = price_data.get("open")

        cash_amount = self.portfolio.get("cash")  # 현금 보유량
        coin_amount = self.portfolio.get(self.coin)  # 코인 보유량
        total_value = cash_amount + coin_amount * price

        # Update portfolio ratio
        self.portfolio_ratio["cash"] = cash_amount / total_value
        self.portfolio_ratio[self.coin] = coin_amount * price / total_value

        # 가치 기록 및 MDD 업데이트
        self._record_value(date, total_value)

    async def update_portfolio_by_trade(
        self,
        price_data: Dict[str, Any],
        coin: CoinType,
        amount: float,
        order_type: OrderType,
    ) -> None:
        """_summary_
        매 거래마다 포트폴리오를 업데이트합니다.
        어떤 자산(현금, 코인)에 대해서 업데이트할 것인지 결정합니다.

        Args:
            price_data (Dict[str, Any]): 가격 데이터
            coin (CoinType): 코인 종류
            amount (float): 주문할 코인의 총 자산 대비 비율
            order_type (OrderType): "buy", "sell" or "hold"

        Raises:
            ValueError: _description_
        """
        if coin not in self.portfolio:
            raise ValueError(f"Coin {coin} not in portfolio.")

        price = price_data["open"]  # 시가

        total_value = self.portfolio["cash"] + self.portfolio[coin] * price  # 총 자산

        if order_type == OrderType.BUY:
            # 수수료가 반영된 매수할 원화 금액
            total_value_after_fee = total_value * amount * (1 - self.fee)
            # 구매할 코인 수량
            pay_amount = total_value_after_fee / price
            # 최종 코인 수량
            self.portfolio[self.coin] += pay_amount
            # 최종 현금 수량, 수수료가 반영되지 않은 현금에서 차감
            self.portfolio["cash"] -= total_value * amount

        elif order_type == OrderType.SELL:
            # 매도할 원화 금액
            sell_value = total_value * amount
            # 판매할 코인 수량
            pay_amount = sell_value / price
            # 최종 코인 수량
            self.portfolio[self.coin] -= pay_amount
            # 최종 현금 수량, 수수료가 반영된 현금에서 차감
            self.portfolio["cash"] += sell_value * (1 - self.fee)

        else:
            pass

        await self.update_portfolio_ratio(price_data=price_data)

    # 모든 코인을 주어진 코인 가격에 맞게 현금으로 판매
    async def sell_all(self, price_data: Dict[str, Any]) -> None:
        """_summary_
        모든 코인을 해당 날의 종가에 판매합니다.

        Args:
            price_data (Dict[str, Any]): 가격 데이터

        Raises:
            ValueError: _description_
        """
        if self.coin not in self.portfolio:
            raise ValueError(f"Coin {self.coin} not in portfolio.")

        price = price_data["close"]
        coin_amount = self.portfolio[self.coin]
        self.portfolio["cash"] += (coin_amount * price) * (1 - self.fee)
        self.portfolio[self.coin] = 0
        await self.update_portfolio_ratio(price_data=price_data, is_sell_all=True)

    def _record_value(self, date: Optional[datetime], current_value: float) -> None:
        # 날짜와 함께 가치 기록
        self.portfolio_value_history.append({"date": date, "value": current_value})
        # 최고치 갱신 및 MDD 계산
        if current_value > self.peak_value:
            self.peak_value = current_value
        drawdown = (self.peak_value - current_value) / self.peak_value * 100
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def get_portfolio(self) -> Dict[str, Any]:
        return self.portfolio

    def get_portfolio_ratio(self) -> Dict[str, float]:
        return self.portfolio_ratio

    def compute_return(self) -> float:
        """전체 수익률 계산"""
        final_value = self.portfolio_value_history[-1]["value"]
        return (final_value - self.initial_value) / self.initial_value * 100

    def compute_mdd(self) -> float:
        """최대 낙폭(MDD) 반환"""
        return self.max_drawdown

    def compute_sharpe(self) -> float:
        """
        샤프 지수 계산 (무위험 수익률 적용)
        interval_minutes 기준으로 연환산 인자를 적용합니다.
        예: 15분봉이라면 interval_minutes=15
        """
        # 가치 시리즈에서 날짜 None 제거 후 값 추출
        values = np.array(
            [
                rec["value"]
                for rec in self.portfolio_value_history
                if rec["date"] is not None
            ]
        )
        if len(values) < 2:
            return 0.0
        # 구간별 수익률 계산
        returns = values[1:] / values[:-1] - 1
        # 초과 수익률
        # 무위험 수익률을 1분 단위로 환산하여 적용
        excess = returns - self.risk_free_rate * (self.interval_minutes / 1440)
        # 표준편차 0 방지
        if excess.std() == 0:
            return 0.0
        # 연환산 인자: (분단위 1년=525600분) / interval_minutes
        periods_per_year = 525600 / self.interval_minutes
        return np.sqrt(periods_per_year) * excess.mean() / excess.std()

    def get_performance(self) -> Dict[str, float]:
        """모든 성과 지표를 계산하여 반환"""
        return {
            "return": self.compute_return(),
            "mdd": self.compute_mdd(),
            "sharpe": self.compute_sharpe(),
        }
