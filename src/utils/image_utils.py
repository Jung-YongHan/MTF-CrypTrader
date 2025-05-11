import io
from typing import Any

import PIL
from autogen_core import Image


def get_agentic_image(fig: Any) -> Image:
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    pil_image = PIL.Image.open(img_buffer)
    return Image(pil_image)
