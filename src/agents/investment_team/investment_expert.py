import json
from os import getenv
from typing import Any, Dict, Literal

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
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
        # 소수점 3자리로 강제
        if not isinstance(v, float):
            raise ValueError("amount must be a float with 3 decimal places")
        parts = str(v).split(".")
        if len(parts) == 3 and len(parts[1]) > 3:
            return float(".".join(parts[:2]))
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


class InvestmentExpert(AssistantAgent):
    def __init__(self):
        self._client = OllamaChatCompletionClient(model="gemma3:27b")
        # self._client = OpenAIChatCompletionClient(
        #     model="gpt-4o-mini", api_key=getenv("OPENAI_API_KEY")
        # )
        super().__init__(
            "investment_expert",
            model_client=self._client,
            output_content_type=OrderTacticianResponse,
            system_message=(
                """당신은 투자 전문가입니다.
주어진 상위 리포트(higher_report), 펄스 리포트(pulse_report), 그리고 현재 포트폴리오 비율(portfolio_ratio)을 바탕으로, 코인(btc)에 대한 적절한 주문(order)과 주문 수량(amount)을 결정하세요.

### 입력 데이터 구조
{
    "higher_report": {
        "trend_report" {
            "trend": str,
            "confidence": float,
            "reason": str
        }
        "limit_report": {
            "rate_limit": float,
            "reason": str
        }
    },
    "lower_report": {
        "pulse_report": {
            "pulse": str,
            "strength": float,
            "reason": str
        },
        "limit_report": {
            "rate_limit": float,
            "reason": str
        }
    }
    "portfolio_ratio": {
        "cash": float,
        "btc": float
    },
}
- higher_report
    - trend_report
        - trend: 거시적 시장 흐름 (ex, 상승장, 하락장, 횡보장)
        - confidence: 추세 분류에 대한 신뢰도 (0.0 ~ 1.0)
        - rate_limit: 코인에 투자 가능한 자산 최대 비율 (0.0 ~ 1.0)
    - limit_report
        - rate_limit: 코인에 투자 가능한 자산 최대 비율 (0.0 ~ 1.0)
        - reason: 비율 선정 이유
- lower_report
    - pulse_report
        - pulse: 단기 시장 신호 (상승 돌파 / 하락 돌파 / 돌파 없음)
        - strength: 해당 신호의 강도 (0.0 ~ 1.0)
    - limit_report
        - rate_limit: 코인에 투자 가능한 자산 최대 비율 (0.0 ~ 1.0)
        - reason: 비율 선정 이유
- portfolio_ratio
    - cash: 현금 비율 (0.0 ~ 1.0)
    - btc: 코인(btc) 비율 (0.0 ~ 1.0)
    - 현금 비율과 코인(btc) 비율의 합은 항상 1.0.


### 출력 데이터 구조
{
    "order": str,
    "amount": float
}
- order: 수행할 주문 종류
    - "buy": 매수
    - "sell": 매도
    - "hold": 보유
- amount: 주문 수량
    - hold일 경우 반드시 0.0

    
### 주문 결정 규칙
1. 매수("buy"):
    - 코인 비중이 rate_limit보다 작을 때만 매수 가능
    - 매수 가능 최대 수량은 rate_limit - 현재 btc 비율
    - 매수하려는 amount는 보유한 cash보다 작거나 같아야 함
2. 매도("sell"):
    - 단순히 btc 비율 > rate_limit인 상황이라도 강제 매도하지 마세요
        - 상승장 또는 강한 상승 돌파 시에는 초과 상태를 유지하며 hold 가능
    - 매도는 오직 거시/미시 리포트가 하락을 강하게 시사할 때만 선택
    - 매도하려는 amount는 보유한 btc보다 작거나 같아야 함
3. 보유("hold"):
    - 상승 시그널이 강하지만 이미 rate_limit을 초과한 경우 → 보유 유지
    - 현금이 부족하여 매수를 못하는 경우 → 보유 유지
    - btc가 없어서 매도를 못하는 경우 → 보유 유지

### 주문 검증 규칙
- 매수 오류:
    - 현금이 0일 때 매수 시도
    - 매수 수량이 보유 현금보다 많을 경우
- 매도 오류:
    - btc가 0일 때 매도 시도
    - 매도 수량이 보유 btc보다 많을 경우

### 예시
- 매수 예
    - 보유 btc: 0.4 / rate_limit: 0.5 -> 최대 매수 가능: 0.1
- 매도 예 (정당한 사유 필요)
    - 보유 btc: 0.5 / rate_limit: 0.3
        - 단, 시장이 약세일 때만 매도 고려
        - 상승 시그널이면 hold 유지 가능

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

                rate_limit = macro_report["limit_report"]["rate_limit"]

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
                            "Sell amount {:.4f} exceeds market balance {:.4f}.".format(
                                amount, coin_ratio
                            )
                        )

                report = content.response.model_dump()
                report["reason"] = thoughts

                await self.close()
                return report
            except ValidationError as e:
                feedback = TextMessage(
                    content=(
                        f"⛔  JSON schema validation failed: {e}\n"
                        "규칙:\n"
                        "1. order가 hold인 경우 amount는 0.0.\n"
                    ),
                    source="validator",
                )
            except ValueError as e:
                feedback = TextMessage(
                    content=(
                        f"⛔  Order rule validation failed: {e}\n"
                        "- sell 시에는 market 비율 이하의 amount만 허용됩니다.\n"
                        "- buy 시에는 cash 비율 이하의 amount만 허용됩니다.\n"
                    ),
                    source="validator",
                )
            messages.append(feedback)
            print(messages)
        print("OrderTactician: 5회 반복 후에도 오류 발생, 보유 유지")
        await self.close()
        return {
            "order": "hold",
            "amount": 0.0,
        }

    async def close(self):
        await self.on_reset(cancellation_token=CancellationToken())
        # await self._client.close()
        # await super().close()
