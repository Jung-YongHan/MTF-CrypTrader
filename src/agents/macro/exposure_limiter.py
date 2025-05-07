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


class ExposureLimiter(AssistantAgent):
    def __init__(self):
        super().__init__(
            "exposure_limiter",
            model_client,
            system_message=(
                "You are ExposureLimiter. "
                "Based on the regime report, output JSON {'exposure_limit': float}."
            ),
        )

    async def limit(self, regime_report: Dict[str, Any]) -> float:
        res = await self.run(task=json.dumps(regime_report))
        return json.loads(res.messages[-1].content)["exposure_limit"]


def get_image() -> PIL.Image.Image:
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    pil_image = PIL.Image.open(img_buffer)
    return Image(pil_image)
