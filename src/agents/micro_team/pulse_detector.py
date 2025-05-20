from os import getenv
from typing import Any, Dict, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage, TextMessage
from autogen_core import CancellationToken
from matplotlib import pyplot as plt
from pydantic import BaseModel, ValidationError, field_validator

from src.config.agent_config import PULSE_DETECTOR_SYSTEM_MESSAGE
from src.enum.pulse_type import PulseType
from src.utils.image_utils import get_agentic_image
from src.utils.model_utils import get_model_client


class PulseReport(BaseModel):
    pulse: PulseType
    strength: float
    reason: Optional[str] = None

    @field_validator("strength")
    @classmethod
    def check_strength(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("strength must be between 0 and 1")
        # 소수점 2자리까지 허용
        if not isinstance(v, float) or len(str(v).split(".")[1]) > 2:
            raise ValueError("strength must be a float with 2 decimal places")
        return v


class PulseDetectorResponse(BaseModel):
    thoughts: str
    response: PulseReport


class PulseDetector(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="pulse_detector",
            model_client=get_model_client(getenv("PULSE_DETECTOR_MODEL")),
            output_content_type=PulseDetectorResponse,
            system_message=PULSE_DETECTOR_SYSTEM_MESSAGE,
        )

    async def detect(self, price_data: Dict[str, Any], fig: Any) -> PulseReport:
        image = get_agentic_image(fig)
        plt.close(fig)

        base_mm = MultiModalMessage(
            content=[image, f"{price_data}"],
            source="data_preprocessor",
        )

        messages = [base_mm]
        while True:  # 최대 5회 반복
            try:
                response = await self.run(task=messages)
                content: PulseDetectorResponse = response.messages[-1].content

                # JSON 스키마 검증
                PulseReport.model_validate(content.response.model_dump())

                pulse_report = content.response
                pulse_report.reason = content.thoughts

                await self.close()
                return pulse_report.model_dump_json()
            except ValidationError as e:  # ← ValidationError 잡기
                feedback = TextMessage(
                    content=(
                        "JSON schema validation failed:"
                        f"{e}\n\n"
                        "규칙:\n"
                        "1. pulse는 '상승 돌파', '하락 돌파', '돌파 없음' 중 하나.\n"
                        " strength는 0.0 ~ 1.0 사이 소수점 두 자리.\n"
                    ),
                    source="validator",
                )
                messages.append(feedback)

    async def close(self):
        await self.on_reset(cancellation_token=CancellationToken())
        # await self._client.close()
        # await super().close()
