from os import getenv
from typing import Any, Dict, Literal

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel, ValidationError

from src.utils.image_utils import get_agentic_image


class PulseResponse(BaseModel):
    pulse: Literal["long", "short", "none"]
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
OHLCV 및 기술적 분석 지표값을 포함한 15분봉 수준의 가격 데이터와 차트 이미지를 분석한 후, 분석 결과를 다음과 같은 JSON 형식으로 출력합니다.

### Output JSON 형식
{pulse: ..., strength: ...}

### 펄스 종류
- long: 매수 신호
- short: 매도 신호
- none: 신호 없음

### 펄스 강도
- (STRICT) 0.0 ~ 1.0 사이 실수값, 소수점 2자리까지 허용
- 0.0: 전혀 신뢰할 수 없는 분석치
- 0.5: 보통 신뢰할 수 있는 분석치
- 1.0: 매우 신뢰할 수 있는 분석치
                            
### 예시
- {pulse: long, strength: 0.8}
- {pulse: short, strength: 0.5}
- {pulse: none, strength: 0.2}
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
        for _ in range(5):  # 최대 5회 반복
            try:
                response = await self.run(task=messages)
                content = response.messages[-1].content
                # pydantic parsing; 범위 벗어나면 error
                PulseResponse.model_validate(content.response.model_dump())

                thoughts = content.thoughts
                pulse_report = content.response

                self.close()
                return pulse_report.dict()
            except ValidationError as e:  # ← ValidationError 잡기
                feedback = TextMessage(
                    content=(
                        "JSON schema validation failed:"
                        f"{e}\n\n"
                        "규칙:\n"
                        "1. pulse는 long, short, none 중 하나.\n"
                        "2. strength는 0.0 ~ 1.0 사이 소수점 두 자리.\n"
                    ),
                    source="validator",
                )
                messages.append(feedback)
        raise RuntimeError("PulseDetector: too many invalid responses")
