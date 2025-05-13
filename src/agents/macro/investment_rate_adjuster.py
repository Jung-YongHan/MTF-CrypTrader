import json
from os import getenv
from typing import Any, Dict

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, ValidationError

from src.portfoilo_manager import PortfolioManager


class RateResponse(BaseModel):
    rate_limit: float

    @pydantic.field_validator("rate_limit")
    @classmethod
    def rate_limit_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("rate_limit must be between 0 and 1")
        # 소수점 2자리까지 허용
        if not isinstance(v, float) or len(str(v).split(".")[1]) > 2:
            raise ValueError("rate_limit must be a float with 2 decimal places")
        return v


class InvestmentRateAdjusterResponse(BaseModel):
    thoughts: str
    response: RateResponse


class InvestmentRateAdjuster(AssistantAgent):
    def __init__(self):
        self._client = OllamaChatCompletionClient(model="gemma3:27b")
        # self._client = OpenAIChatCompletionClient(
        #     model="gpt-4o-mini", api_key=getenv("OPENAI_API_KEY")
        # )
        super().__init__(
            "investment_rate_adjuster",
            model_client=self._client,
            output_content_type=InvestmentRateAdjusterResponse,
            system_message=(
                """당신은 투자 비율 조정가입니다.
주어진 시장 분석 보고서(regime_report), 가격 데이터(price_data), 그리고 현재 포트폴리오 비율(portfolio_ratio)를 기반으로, 코인 자산에 대한 최대 투자 비율(rate_limit)을 결정해야 합니다.

### 입력 데이터 구조
{
    "regime_report": {
        "regime": "상승장" | "하락장" | "횡보장",
        "confidence": 0.0 ~ 1.0,
        "reason": "regime 판단 이유"
    },
    "price_data": {
        "timestamp": "2025-01-01 09:00:00",
        "open": 10000,
        "high": 11000,
        "low": 9000,
        "close": 10500,
        "volume": 1000,
        ...
    },
    "portfolio_ratio": {
        "cash": 0.6,
        "btc": 0.4
    }
}

- regime_report: 시장의 현재 상태와 그에 대한 확신도 및 근거를 나타냅니다.
- price_data: OHLCV 및 기타 기술적 지표를 포함한 일일 가격 데이터입니다.
- portfolio_ratio: 현재 포트폴리오에서 현금과 코인 자산의 비율을 나타냅니다.

### 출력 데이터 구조
{
    "rate_limit": 0.0 ~ 1.0
}
- rate_limit: 코인 자산에 대한 투자 비율을 나타내는 0.0에서 1.0 사이의 실수값입니다. 소수점은 최대 두 자리까지 허용됩니다.

### 투자 비율 정의
- 0.0: 전체 자산을 현금으로 보유 (코인 투자 없음)
- 0.5: 전체 자산의 50%를 코인에 투자
- 1.0: 전체 자산을 코인에 투자

### 결정 지침
1. **시장 상태(regime)**와 **확신도(confidence)**를 고려하여 적절한 최대 투자 비율을 결정합니다.
2. **가격 데이터(price_data)**의 기술적 지표를 분석하여 시장의 변동성과 추세를 평가합니다.
3. 상승장에서는 최대 투자 비율을 높이고, 하락장 및 고변동성장에서는 낮추며, 횡보장에서는 중립적으로 유지합니다.
4. **현재 포트폴리오 비율(portfolio_ratio)**을 참고하여 투자 비율을 조정합니다.
5. 최종 결정된 투자 비율은 반드시 0.0에서 1.0 사이의 실수값이어야 하며, 소수점은 최대 두 자리까지 허용됩니다.

### 예시
{ "rate_limit": 0.5 }
"""
            ),
        )

    async def adjust_rate_limit(
        self,
        regime_report: Dict[str, Any],
        price_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        report = {
            "regime_report": regime_report,
            "price_data": price_data,
            "portfolio_ratio": PortfolioManager.get_instance().get_portfolio_ratio(),
        }
        base_msg = TextMessage(
            content=json.dumps(report, indent=4),
            source="data_preprocessor",
        )
        message = [base_msg]

        while True:  # 최대 5회 반복
            try:
                response = await self.run(task=message)
                content = response.messages[-1].content
                RateResponse.model_validate({"rate_limit": content.response.rate_limit})

                thoughts = content.thoughts
                rate_limit = content.response.rate_limit

                report = {
                    "regime_report": regime_report,
                    "limit_report": {
                        "rate_limit": rate_limit,
                        "reason": thoughts,
                    },
                }

                await self.close()
                return report
            except ValidationError as e:  # ← ValidationError 잡기
                feedback = TextMessage(
                    content=(
                        "JSON schema validation failed:"
                        f"{e}\n\n"
                        "규칙:\n"
                        "- rate_limit은 0.0 ~ 1.0 사이 소수점 두 자리.\n"
                    ),
                    source="validator",
                )
                message.append(feedback)

    async def close(self):
        await self.on_reset(cancellation_token=CancellationToken())
        # await self._client.close()
        # await super().close()
