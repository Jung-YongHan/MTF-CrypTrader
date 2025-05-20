from os import getenv
from typing import Any, Dict, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import CancellationToken
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, field_validator

from src.config.agent_config import REGIME_ANALYZER_SYSTEM_MESSAGE
from src.enum.regime_type import RegimeType
from src.utils.image_utils import get_agentic_image
from src.utils.model_utils import get_model_client


class RegimeReport(BaseModel):
    regime: RegimeType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def check_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v


class RegimeAnalyzerResponse(BaseModel):
    thoughts: str
    response: RegimeReport


class RegimeAnalyzer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="regime_analyzer",
            model_client=get_model_client(getenv("REGIME_ANALYZER_MODEL")),
            output_content_type=RegimeAnalyzerResponse,
            system_message=REGIME_ANALYZER_SYSTEM_MESSAGE,
        )

    async def analyze(self, price_data: Dict[str, Any], fig: Any) -> RegimeReport:
        image = get_agentic_image(fig)
        plt.close(fig)

        message = MultiModalMessage(
            content=[image, f"{price_data}"],
            source="data_preprocessor",
        )

        response = await self.run(task=[message])
        content: RegimeAnalyzerResponse = response.messages[-1].content

        regime_report = content.response
        regime_report.reason = content.thoughts

        await self.close()
        return regime_report

    async def close(self):
        await self.on_reset(cancellation_token=CancellationToken())
        # await self._client.close()
        # await super().close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    regime_analyzer = RegimeAnalyzer()
