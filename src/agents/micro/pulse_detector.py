import json
from os import getenv
from typing import Any, Dict

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


class PulseDetector(AssistantAgent):
    def __init__(self):
        super().__init__(
            "pulse_detector",
            model_client=OllamaChatCompletionClient(
                model=getenv("PULSE_DETECTOR_MODEL")
            ),
            system_message=(
                "You are PulseDetector. "
                "Analyze minute OHLCV plus macro_report JSON and detect trading pulse. "
                "Return JSON {pulse: 'long'|'short'|'none', strength: float}."
            ),
        )

    async def detect(
        self, minute_data: Dict[str, Any], macro_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = {"minute": minute_data, "macro_report": macro_report}
        res = await self.run(task=json.dumps(payload))
        return json.loads(res.messages[-1].content)
