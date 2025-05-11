from os import getenv
from typing import Any, Dict, Literal

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel

from src.utils.image_utils import get_agentic_image


class RegimeReport(BaseModel):
    regime: Literal["bull", "bear", "sideways", "high_volatility"]
    confidence: float

    @pydantic.field_validator("confidence")
    @classmethod
    def confidence_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


class RegimeAnalyzerResponse(BaseModel):
    thoughts: str
    response: RegimeReport


class RegimeAnalyzer(AssistantAgent):
    def __init__(self):
        super().__init__(
            "regime_analyzer",
            model_client=OllamaChatCompletionClient(
                model=getenv("REGIME_ANALYZER_MODEL")
            ),
            output_content_type=RegimeAnalyzerResponse,
            system_message=(
                """당신은 시장 레짐 분석가입니다.
OHLCV 및 기술적 분석 지표 값을 포함한 일일 가격 데이터와 차트 이미지를 분석한 후, 분석 결과를 다음과 같은 JSON 형식으로 출력합니다.

### Output JSON 형식
{regime: ..., confidence: ...}.

### 레짐 종류
- bull: 상승장
- bear: 하락장
- sideways: 횡보장
- high_volatility: 고변동성장

### 신뢰도 구간
- (STRICT) 0.0 ~ 1.0 사이 실수값
- 0.0: 전혀 신뢰할 수 없는 분석치
- 1.0: 매우 신뢰할 수 있는 분석치

### 예시
- {regime: bull, confidence: 0.8}
- {regime: bear, confidence: 0.5}
- {regime: range, confidence: 0.2}
- {regime: high_volatility, confidence: 0.9}
"""
            ),
        )

    async def analyze(self, price_data: Dict[str, Any], fig: Any) -> Dict[str, Any]:
        image = get_agentic_image(fig)
        message = MultiModalMessage(
            content=[image, f"{price_data}"],
            source="data_preprocessor",
        )

        response = await self.run(task=[message])
        content = response.messages[-1].content

        thoughts = content.thoughts
        regime_report = content.response

        self.close()
        return regime_report.dict()
