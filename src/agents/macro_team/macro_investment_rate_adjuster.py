import json
from os import getenv
from typing import Any, Dict, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.agents.macro_team.regime_analyzer import RegimeReport
from src.config.agent_config import MACRO_INVESTMENT_RATE_ADJUSTER_SYSTEM_MESSAGE
from src.portfoilo_manager import PortfolioManager
from src.utils.model_utils import get_model_client


class LimitReport(BaseModel):
    rate_limit: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None

    @field_validator("rate_limit")
    @classmethod
    def check_rate_limit(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("rate_limit must be between 0 and 1")
        # 소수점 2자리까지 허용
        if not isinstance(v, float) or len(str(v).split(".")[1]) > 2:
            raise ValueError("rate_limit must be a float with 2 decimal places")
        return v


class MacroInvestmentRateAdjusterResponse(BaseModel):
    thoughts: str
    response: LimitReport


class MacroInvestmentRateAdjuster(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="investment_rate_adjuster",
            model_client=get_model_client(
                getenv("MACRO_INVESTMENT_RATE_ADJUSTER_MODEL")
            ),
            output_content_type=MacroInvestmentRateAdjusterResponse,
            system_message=MACRO_INVESTMENT_RATE_ADJUSTER_SYSTEM_MESSAGE,
        )

    async def adjust_rate_limit(
        self,
        regime_report: RegimeReport,
        price_data: Dict[str, Any],
    ) -> LimitReport:
        prompt = {
            "regime_report": regime_report,
            "price_data": price_data,
            "portfolio_ratio": PortfolioManager.get_instance().get_portfolio_ratio(),
        }
        base_msg = TextMessage(
            content=json.dumps(prompt, indent=4),
            source="data_preprocessor",
        )

        message = [base_msg]
        while True:  # 최대 5회 반복
            try:
                response = await self.run(task=message)
                content: MacroInvestmentRateAdjusterResponse = response.messages[
                    -1
                ].content

                # JSON 스키마 검증
                LimitReport.model_validate({"rate_limit": content.response.rate_limit})

                limit_report = content.response
                limit_report.reason = content.thoughts

                await self.close()
                return limit_report.model_dump_json()
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    rate_adjuster = MacroInvestmentRateAdjuster()
