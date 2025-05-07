import json
from typing import Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient

model_client = OllamaChatCompletionClient(model="gemma3:4b")


class TradeExecutor(AssistantAgent):
    def __init__(self):
        super().__init__(
            "trade_executor",
            model_client,
            system_message=(
                "You are TradeExecutor. "
                "Execute trades based on order_report JSON. "
                "Return JSON {'status': 'filled'|'rejected', 'avg_price': float, 'filled_qty': float}."
            ),
        )

    async def execute(self, order_report: Dict[str, Any]) -> Dict[str, Any]:
        res = await self.run(task=json.dumps(order_report))
        return json.loads(res.messages[-1].content)
