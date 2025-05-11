import json
from os import getenv
from typing import Any, Dict

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel, ValidationError

from src.agents.portfoilo_manager import PortfolioManager


class ExposureResponse(BaseModel):
    exposure: float

    @pydantic.field_validator("exposure")
    @classmethod
    def exposure_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("exposure must be between 0 and 1")
        # 소수점 2자리까지 허용
        if not isinstance(v, float) or len(str(v).split(".")[1]) > 2:
            raise ValueError("exposure must be a float with 2 decimal places")
        return v


class RegimeAnalyzerResponse(BaseModel):
    thoughts: str
    response: ExposureResponse


class ExposureLimiter(AssistantAgent):
    def __init__(self):
        super().__init__(
            "exposure_limiter",
            model_client=OllamaChatCompletionClient(
                model=getenv("EXPOSURE_LIMITER_MODEL")
            ),
            output_content_type=RegimeAnalyzerResponse,
            system_message=(
                """당신은 노출 한도 관리자입니다.
regime_report 및 price_data(OHLCV 및 기술적 분석 지표)와 현재 portfolio ratio 현황을 기반으로 자산 투자 한도를 제한해야 합니다.
이때 제한하는 자산은 현금을 제외한 코인 자산에 대해서 제한합니다.
예시로, 자산 투자 한도가 0.5인 경우, 자산의 절반을 코인에 투자할 수 있습니다.

### 입력 JSON 형식
{
    regime_report: {
        regime: bull | bear | sideways | high_volatility,
        confidence: 0‑1,   // regime classifier confidence
    },
    price_data: {
        "timestamp": 2025-01-01 09:00:00,
        "open": 10000,
        "high": 11000,
        "low": 9000,
        "close": 10500,
        "volume": 1000,
        ...
    },
    portfolio_ratio: {
        "cash": 0.6,
        "btc": 0.4
    },
}

### 출력 JSON 형식
{exposure_limit: ...} 

### 노출 제한 구간
- (STRICT) 0 ~ 1 사이 실수값, 소수점 2자리까지 허용
- 0: 자산의 전부를 현금으로 보유
- 0.5: 자산의 절반을 코인으로 보유
- 1: 자산의 전부를 코인으로 투자

### 예시
- {exposure_limit: 0.5}
- {exposure_limit: 0.2}
- {exposure_limit: 0.8}
"""
            ),
        )

    async def limit_exposure(
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

        for _ in range(5):  # 최대 5회 반복
            try:
                response = await self.run(task=message)
                content = response.messages[-1].content
                ExposureResponse.model_validate({"exposure": content.response.exposure})

                thoughts = content.thoughts
                exposure_limit = content.response.exposure

                regime_report_including_exposure = regime_report.copy()
                regime_report_including_exposure["exposure_limit"] = exposure_limit

                self.close()
                return regime_report_including_exposure
            except ValidationError as e:  # ← ValidationError 잡기
                feedback = TextMessage(
                    content=(
                        "JSON schema validation failed:"
                        f"{e}\n\n"
                        "규칙:\n"
                        "- exposure는 0.00~1.00 사이 소수점 두 자리.\n"
                    ),
                    source="validator",
                )
                message.append(feedback)
        raise RuntimeError("ExposureLimiter: too many invalid responses")
