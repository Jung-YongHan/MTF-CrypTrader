import json
from os import getenv
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel


class BetaResponse(BaseModel):
    beta: float


class RegimeAnalyzerResponse(BaseModel):
    thoughts: str
    response: BetaResponse


class BetaStrategist(AssistantAgent):
    def __init__(self):
        super().__init__(
            "beta_strategist",
            model_client=OllamaChatCompletionClient(
                model=getenv("BETA_STRATEGIST_MODEL")
            ),
            system_message=(
                """당신은 베타 전략가입니다.
시장 레짐 분석 결과와 현재 포트폴리오를 바탕으로 다음 투자를 위한 적절한 목표 포트폴리오 베타 값을 결정합니다.
### 입력 JSON 형식
{
    regime: bull | bear | sideways | high_volatility,
    confidence: 0‑1   // regime classifier confidence
}

### 베타 값 예시


"""
            ),
        )

    async def select(self, regime_report: Dict[str, Any]) -> float:
        res = await self.run(task=json.dumps(regime_report))
        return json.loads(res.messages[-1].content)["beta"]
