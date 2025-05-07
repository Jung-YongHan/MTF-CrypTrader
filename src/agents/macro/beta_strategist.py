import io
import json
from typing import Any, Dict, List, Literal

import PIL
import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel

model_client = OllamaChatCompletionClient(model="gemma3:4b")


class BetaStrategist(AssistantAgent):
    def __init__(self):
        super().__init__(
            "beta_strategist",
            model_client,
            system_message=(
                "You are BetaStrategist. "
                "Based on the regime report, output JSON {'beta': float}."
            ),
        )

    async def select(self, regime_report: Dict[str, Any]) -> float:
        res = await self.run(task=json.dumps(regime_report))
        return json.loads(res.messages[-1].content)["beta"]
