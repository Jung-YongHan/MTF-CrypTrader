import json
from os import getenv
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


class FeedbackReflector(AssistantAgent):
    def __init__(self):
        super().__init__(
            "feedback_reflector",
            model_client=OllamaChatCompletionClient(
                model=getenv("FEEDBACK_REFLECTOR_MODEL")
            ),
            system_message=(
                "You are FeedbackReflector. "
                "Based on trade_summary JSON (order + result), "
                "output JSON with keys: 'metrics', 'improvements', 'patch'."
            ),
        )

    async def reflect(self, trade_summary: Dict[str, Any]) -> Dict[str, Any]:
        res = await self.run(task=json.dumps(trade_summary))
        return json.loads(res.messages[-1].content)
