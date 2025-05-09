import json
from os import getenv
from typing import Any, Dict, List

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel


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
regime 및 가격 데이터(OHLCV 및 기술적 분석 지표)를 바탕으로 포트폴리오의 투자 한도를 제한해야 합니다.
이때, 현재 포트폴리오의 현황도 함께 고려해야 합니다.

### 입력 JSON 형식
{
    regime: bull | bear | sideways | high_volatility,
    confidence: 0‑1,   // regime classifier confidence
    portfolio: {
        "cash": 0.6,
        "btc": 0.4
    },
    price_data: [
        {
            "timestamp": 2025-01-01 09:00:00,
            "open": 10000,
            "high": 11000,
            "low": 9000,
            "close": 10500,
            "volume": 1000,
            ...
        },
        ...
    ],

}

### 출력 JSON 형식
{exposure_limit: ...} 

### 노출 제한 구간
- (STRICT) 0 ~ 1 사이 실수값, 소수점 2자리까지 허용
- 0: 자산의 전부를 현금으로 보유
- 0.5: 자산의 절반을 현금으로 보유
- 1: 자산의 전부를 투자

### 예시
- {exposure_limit: 0.5}
- {exposure_limit: 0.2}
- {exposure_limit: 0.8}
"""
            ),
        )

    async def limit_exposure(
        self,
        portfolio: Dict[str, Any],
        regime_report: Dict[str, Any],
        price_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        report = {**portfolio, **regime_report, "price_data": price_data}
        report = {**portfolio, **regime_report, "price_data": price_data}
        while True:
            content = None
            try:
                response = await self.run(task=json.dumps(report))
                content = response.messages[-1].content
                # pydantic parsing; 범위 벗어나면 error
                ExposureResponse.model_validate({"exposure": content.response.exposure})
                break
            except ValueError:
                # 범위 벗어남 → 다시 요청
                if content is not None and hasattr(content, "response"):
                    print(
                        f"Exposure limit out of range: {content.response.exposure}. Retrying..."
                    )
                else:
                    print(
                        "Exposure limit out of range and content is None. Retrying..."
                    )
                continue

        thoughts = content.thoughts
        exposure_limit = content.response.exposure

        regime_report_including_exposure = regime_report.copy()
        regime_report_including_exposure["exposure_limit"] = exposure_limit
        return regime_report_including_exposure
