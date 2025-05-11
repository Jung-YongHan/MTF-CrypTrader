from os import getenv
from typing import Any, Dict, Literal

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient
from matplotlib import pyplot as plt
from pydantic import BaseModel, ValidationError

from src.utils.image_utils import get_agentic_image


class PulseResponse(BaseModel):
    pulse: Literal["매수 돌파", "매도 돌파", "돌파 없음"]
    strength: float

    @pydantic.field_validator("strength")
    @classmethod
    def strength_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        # 소수점 2자리까지 허용
        if not isinstance(v, float) or len(str(v).split(".")[1]) > 2:
            raise ValueError("strength must be a float with 2 decimal places")
        return v


class PulseDetectorResponse(BaseModel):
    thoughts: str
    response: PulseResponse


class PulseDetector(AssistantAgent):
    def __init__(self):
        super().__init__(
            "pulse_detector",
            model_client=OllamaChatCompletionClient(
                model=getenv("PULSE_DETECTOR_MODEL")
            ),
            output_content_type=PulseDetectorResponse,
            system_message=(
                """당신은 펄스 감지자입니다.
주어진 15분봉 수준의 가격 데이터(OHLCV 및 기술적 분석 지표)와 차트 이미지를 분석하여, 현재 시장의 돌파(pulse) 여부와 그 강도(strength)를 판단하고, 아래의 JSON 형식으로 결과를 출력해야 합니다.

### 입력 데이터 구조
- 가격 데이터: 15분봉 수준의 OHLCV 및 기술적 지표를 포함한 JSON 형식의 데이터
- 차트 이미지: 해당 가격 데이터를 시각화한 차트 이미지

### 출력 JSON 형식
{
    "pulse": "매수 돌파" | "매도 돌파" | "돌파 없음",
    "strength": 0.0 ~ 1.0
}

- pulse: 감지된 돌파 신호의 종류를 나타냅니다.
    1. "매수 돌파": 상승 방향의 돌파 신호
    2. "매도 돌파": 하락 방향의 돌파 신호
    3. "돌파 없음": 유의미한 돌파 신호 없음
- strength: 감지된 돌파 신호의 강도를 나타내는 0.0에서 1.0 사이의 실수값입니다. 소수점은 최대 두 자리까지 허용됩니다.
    - 0.0부터 1.0 사이의 실수값
    - 0.0: 전혀 신뢰할 수 없는 분석치
    - 0.5: 보통 수준의 신뢰도
    - 1.0: 매우 높은 신뢰도

### 예시
{ "pulse": "매수 돌파", "strength": 0.8 }
{ "pulse": "매도 돌파", "strength": 0.5 }
{ "pulse": "돌파 없음", "strength": 0.2 }
    """
            ),
        )

    async def detect(self, price_data: Dict[str, Any], fig: Any) -> Dict[str, Any]:
        image = get_agentic_image(fig)
        base_mm = MultiModalMessage(
            content=[image, f"{price_data}"],
            source="data_preprocessor",
        )
        messages = [base_mm]
        while True:  # 최대 5회 반복
            try:
                response = await self.run(task=messages)
                content = response.messages[-1].content
                # pydantic parsing; 범위 벗어나면 error
                PulseResponse.model_validate(content.response.model_dump())

                thoughts = content.thoughts
                pulse_report = content.response

                await self.on_reset(cancellation_token=CancellationToken())
                plt.close(fig)
                return pulse_report.dict()
            except ValidationError as e:  # ← ValidationError 잡기
                feedback = TextMessage(
                    content=(
                        "JSON schema validation failed:"
                        f"{e}\n\n"
                        "규칙:\n"
                        "1. pulse는 '매수 돌파', '매도 돌파', '돌파 없음' 중 하나.\n"
                        " strength는 0.0 ~ 1.0 사이 소수점 두 자리.\n"
                    ),
                    source="validator",
                )
                messages.append(feedback)
