import json
from os import getenv
from typing import Any, Dict, Literal

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel, ValidationError

from src.portfoilo_manager import PortfolioManager


class OrderResponse(BaseModel):
    order: Literal["buy", "sell", "hold"]
    amount: float

    @pydantic.field_validator("amount")
    @classmethod
    def amount_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("amount must be between 0 and 1")
        # 소수점 2자리까지 허용
        if not isinstance(v, float):
            raise ValueError("amount must be a float with 2 decimal places")
        parts = str(v).split(".")
        if len(parts) == 2 and len(parts[1]) > 2:
            raise ValueError("amount must be a float with 2 decimal places")
        return v

    @pydantic.model_validator(mode="after")
    def fix_hold_amount(self):
        if self.order == "hold":
            # 모델이 실수해도 amount를 0.0으로 덮어씀
            return self.model_copy(update={"amount": 0.0})
        return self


class OrderTacticianResponse(BaseModel):
    thoughts: str
    response: OrderResponse


class OrderTactician(AssistantAgent):
    def __init__(self):
        self._client = OllamaChatCompletionClient(model="gemma3:4b")
        super().__init__(
            "order_tactician",
            model_client=self._client,
            output_content_type=OrderTacticianResponse,
            system_message=(
                """당신은 주문 전술가입니다.
주어진 매크로 리포트(macro_report), 펄스 리포트(pulse_report), 그리고 현재 포트폴리오 비율(portfolio_ratio)을 코인에 대한 적절한 주문(order)과 주문 수량(amount)을 결정해야 합니다.

### 입력 데이터 구조
{
    "macro_report": {
        "regime": "bull" | "bear" | "sideways" | "high_volatility",
        "confidence": 0.0 ~ 1.0,
        "rate_limit": 0.0 ~ 1.0,
    },
    "pulse_report": {
        "pulse": "long" | "short" | "none",
        "strength": 0.0 ~ 1.0,
    },
    "portfolio_ratio": {
        "cash": float,
        "btc": float
    },
}

- macro_report: 일봉 데이터를 기반으로 한 시장 분석 리포트입니다.
    1. regime: 시장의 레짐 상태를 나타냅니다.
    2. confidence: 레짐 분류에 대한 확신도입니다.
    3. rate_limit: 전체 자산 중 코인에 투자 가능한 최대 비율입니다.
- pulse_report: 15분봉 데이터를 기반으로 한 단기 신호 리포트입니다.
    1. pulse: 돌파 신호의 종류를 나타냅니다.
    2. strength: 돌파 신호의 강도입니다.
- portfolio_ratio: 현재 포트폴리오 내 자산 비율을 나타냅니다.
    예시: {"cash": 0.6, "btc": 0.4}

### 출력 데이터 구조
{
    "order": "buy" | "sell" | "hold",
    "amount": float
}

- order: 수행할 주문의 종류입니다.
    1. "buy": 매수 주문
    2. "sell": 매도 주문
    3. "hold": 보유 유지
- amount: 주문 수량을 나타내는 0.0 ~ 1.0 사이의 실수값입니다.
    소수점은 최대 두 자리까지 허용됩니다.
    "hold" 주문의 경우, amount는 반드시 0.0이어야 합니다.

### 주문 결정 규칙
- 보유 비율이 0인 경우: 해당 코인에 대한 매도("sell") 주문은 불가능합니다.
- 최대 주문 한도: rate_limit 값은 해당 코인에 투자 가능한 최대 비율을 의미합니다. 기존 보유 비율을 고려하여, 추가 매수 또는 매도 가능한 최대 수량을 계산해야 합니다.
- 이때, 시장 상황과 펄스 신호를 신중하게 고려하여 매수("buy") 또는 매도("sell") 주문을 결정합니다.
- 강제 매도 조건: 현재 보유 비율이 rate_limit 값을 초과하는 경우, 초과분에 대해 매도 주문을 실행해야 합니다.
- 보유 유지 조건: 현재 보유 비율이 rate_limit 값과 동일한 경우, 추가 매수 없이 보유를 유지해야 합니다.
- 주문 수량 계산: 주문 수량은 rate_limit과과 현재 보유 비율의 차이로 계산됩니다.

### 추가 검증 규칙
- 매도 오류:
    1. portfolio_ratio의 코인 비율이 0일 때, 매도("sell") 주문을 시도한 경우
    2. 코인 비율이 존재하지만, 매도 주문의 amount가 코인 비율보다 큰 경우
    → 매도 시에는 코인 비율 이하의 amount만 허용됩니다.
- 매수 오류:
    1. portfolio_ratio의 현금 비율이 0일 때, 매수("buy") 주문을 시도한 경우
    2. 현금 비율이 존재하지만, 매수 주문의 amount가 현금 비율보다 큰 경우
    → 매수 시에는 현금 비율 이하의 amount만 허용됩니다.

- 노출 한도 초과 오류:
    1. 현재 코인 비율이 rate_limit 값을 초과하는 경우
    → 초과분에 대해 매도 주문을 실행해야 합니다.

### 예시
- 현재 보유 비율이 0.4이고, rate_limit이 0.5인 경우: 최대 매수 수량은 0.1
- 현재 보유 비율이 0.4이고, rate_limit이 0.3인 경우: 최소 매도 수량은 0.1

### 예시 출력
{ "order": "buy", "amount": 0.5 }
{ "order": "sell", "amount": 0.2 }
{ "order": "hold", "amount": 0.0 }
"""
            ),
        )

    async def decide(
        self,
        macro_report: Dict[str, Any],
        pulse_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        report = {
            "macro_report": macro_report,
            "pulse_report": pulse_report,
            "portfolio_ratio": PortfolioManager.get_instance().get_portfolio_ratio(),
        }
        base_msg = TextMessage(
            content=json.dumps(report, indent=4),
            source="data_preprocessor",
        )
        messages = [base_msg]
        for _ in range(5):  # 최대 5회 반복
            try:
                response = await self.run(task=messages)
                content = response.messages[-1].content
                # pydantic parsing; 범위 벗어나면 error
                OrderResponse.model_validate(content.response.model_dump())

                thoughts = content.thoughts

                order = content.response.order
                amount = content.response.amount
                ratios = report["portfolio_ratio"]

                cash_ratio = ratios.get("cash", 0.0)
                coin_ratio = sum(v for k, v in ratios.items() if k != "cash")

                rate_limit = macro_report.get("rate_limit", 0.0)

                # 추가 검증: buy 오류
                if order == "buy":
                    if round(amount, 4) > round(rate_limit, 4):
                        raise ValueError(
                            "Buy amount {:.4f} exceeds rate_limit limit {:.4f}.".format(
                                amount, rate_limit
                            )
                        )
                    # 소수점 4자리까지 반올림하여 비교
                    if round(coin_ratio + amount, 4) > round(rate_limit, 4):
                        raise ValueError(
                            "Buy amount {:.4f} exceeds rate_limit limit {:.4f}.".format(
                                amount, rate_limit
                            )
                        )

                # 추가 검증: sell 오류
                if order == "sell":
                    if round(amount, 4) > round(coin_ratio, 4):
                        raise ValueError(
                            "Sell amount {:.4f} exceeds coin balance {:.4f}.".format(
                                amount, coin_ratio
                            )
                        )

                # 추가 검증: 코인 비율이 노출 한도를 초과하는 경우
                if round(coin_ratio, 4) > round(rate_limit, 4):
                    # 오차 1% (0.01)까지는 허용
                    if coin_ratio - rate_limit > 0.01:
                        raise ValueError(
                            f"현재 코인 보유 비율({coin_ratio:.4f})이 노출 한도({rate_limit:.4f})를 1% 이상 초과합니다. "
                            "초과된 비율만큼 매도 주문을 생성해야 합니다."
                        )

                await self.close()
                return content.response.model_dump()
            except ValidationError as e:
                feedback = TextMessage(
                    content=(
                        f"⛔  JSON schema validation failed: {e}\n"
                        "규칙:\n"
                        "1. order가 hold인 경우 amount는 0.0.\n"
                        "2. amount는 0.0 ~ 1.0 사이 실수값, 소수점 2자리까지 허용\n"
                    ),
                    source="validator",
                )
            except ValueError as e:
                feedback = TextMessage(
                    content=(
                        f"⛔  Order rule validation failed: {e}\n"
                        "- sell 시에는 coin 비율 이하의 amount만 허용됩니다.\n"
                        "- buy 시에는 cash 비율 이하의 amount만 허용됩니다.\n"
                        "- 현재 코인 보유 비율이 노출 한도를 초과하는 경우, 초과된 비율만큼 매도 주문을 생성해야 합니다."
                    ),
                    source="validator",
                )
            messages.append(feedback)
        print("OrderTactician: 5회 반복 후에도 오류 발생, 보유 유지")
        await self.close()
        return {
            "order": "hold",
            "amount": 0.0,
        }

    async def close(self):
        await self.on_reset(cancellation_token=CancellationToken())
        await self._client.close()
        await super().close()
