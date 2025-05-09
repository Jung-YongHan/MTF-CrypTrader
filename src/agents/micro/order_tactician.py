import json
from os import getenv
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


class OrderTactician(AssistantAgent):
    def __init__(self):
        super().__init__(
            "order_tactician",
            model_client=OllamaChatCompletionClient(
                model=getenv("ORDER_TACTICIAN_MODEL")
            ),
            system_message=(
                "You are OrderTactician. "
                "Given pulse_report JSON and portfolio JSON, "
                "choose order_type ('limit'|'market'|'twap') and qty. "
                "Return JSON {order_type: ..., price: ..., qty: ...}."
            ),
        )

    async def decide(
        self, pulse_report: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = {"pulse_report": pulse_report, "portfolio": portfolio}
        res = await self.run(task=json.dumps(payload))
        return json.loads(res.messages[-1].content)
