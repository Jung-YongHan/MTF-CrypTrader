import os
from typing import Union

from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient


def get_model_client(
    model_name: str,
) -> Union[OpenAIChatCompletionClient, OllamaChatCompletionClient]:
    """
    모델 이름에 따라 적절한 모델 클라이언트를 반환합니다.
    """
    if model_name.startswith("gpt") or model_name.startswith("o"):
        return OpenAIChatCompletionClient(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        return OllamaChatCompletionClient(model=model_name)


if __name__ == "__main__":
    # 테스트용 코드
    model_name = "gemma3:27b"  # 예시 모델 이름
    client = get_model_client(model_name)
    print(f"Model client for {model_name}: {client}")
