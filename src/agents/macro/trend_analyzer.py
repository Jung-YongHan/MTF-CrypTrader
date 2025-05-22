from os import getenv
from typing import Any, Dict, Literal

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from matplotlib import pyplot as plt
from pydantic import BaseModel

from src.utils.image_utils import get_agentic_image


class TrendReport(BaseModel):
    trend: Literal["상승장", "하락장", "횡보장", "고변동성장"]
    confidence: float

    @pydantic.field_validator("confidence")
    @classmethod
    def confidence_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


class TrendAnalyzerResponse(BaseModel):
    thoughts: str
    response: TrendReport


class TrendAnalyzer(AssistantAgent):
    def __init__(self):
        self._client = OllamaChatCompletionClient(model="gemma3:27b")
        # self._client = OpenAIChatCompletionClient(
        #     model="gpt-4o-mini", api_key=getenv("OPENAI_API_KEY")
        # )
        super().__init__(
            "trend_analyzer",
            model_client=self._client,
            output_content_type=TrendAnalyzerResponse,
            system_message=(
                """당신은 시장 추세 분석가입니다.
OHLCV 데이터와 관련 기술 지표 값, 그리고 캔들 차트 이미지를 기반으로 현재 시장의 추세를 분석한 후, 아래의 JSON 형식으로 결과를 출력해야 합니다.

### 입력 데이터 구조
- 가격 데이터: OHLCV 및 기술 지표를 포함한 JSON 형식의 데이터
- 캔들 차트 이미지: 가격 데이터를 시각화한 캔들 차트 이미지

### 출력 JSON 형식
{
    "trend": "상승장" | "하락장" | "횡보장",
    "confidence": 0.0 ~ 1.0
}

### 추세 유형
- 상승장: 가격이 뚜렷한 상승 추세를 보이는 시장
- 하락장: 가격이 명확한 하락 추세를 보이는 시장
- 횡보장: 가격이 뚜렷한 방향 없이 횡보하는 시장

### 신뢰도(confidence)
- 0.0부터 1.0 사이의 실수값
- 0.0: 전혀 신뢰할 수 없음
- 1.0: 매우 높은 신뢰도

### 예시
- { "trend": "상승장", "confidence": 0.8 }
- { "trend": "하락장", "confidence": 0.5 }
- { "trend": "횡보장", "confidence": 0.2 }
"""
            ),
        )

    async def analyze(self, price_data: Dict[str, Any], fig: Any) -> Dict[str, Any]:
        image = get_agentic_image(fig)
        plt.close(fig)

        message = MultiModalMessage(
            content=[image, f"{price_data}"],
            source="data_preprocessor",
        )

        response = await self.run(task=[message])
        content = response.messages[-1].content

        thoughts = content.thoughts
        trend_report = content.response

        report = trend_report.dict()
        report["reason"] = thoughts

        await self.close()
        return report

    async def close(self):
        await self.on_reset(cancellation_token=CancellationToken())
        # await self._client.close()
        # await super().close()
