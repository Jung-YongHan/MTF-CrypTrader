import json
from os import getenv
from typing import Any, Dict, Literal

import pydantic
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from pydantic import BaseModel, ValidationError

from src.agents.portfoilo_manager import PortfolioManager


class OrderResponse(BaseModel):
    order: Literal["buy", "sell", "hold"]
    amount: float

    @pydantic.field_validator("amount")
    @classmethod
    def amount_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("amount must be between 0 and 1")
        # 소수점 2자리까지 허용
        if not isinstance(v, float) or len(str(v).split(".")[1]) > 2:
            raise ValueError("amount must be a float with 2 decimal places")
        return v

    @pydantic.model_validator(mode="after")
    def fix_hold_amount(self):
        if self.order == "hold":
            # 모델이 실수해도 amount를 0.0으로 덮어씀
            return self.model_copy(update={"amount": 0.0})
        return self


class OrderTacticianResponse(BaseModel):
    thoughts: str
    response: OrderResponse


class OrderTactician(AssistantAgent):
    def __init__(self):
        super().__init__(
            "order_tactician",
            model_client=OllamaChatCompletionClient(
                model=getenv("ORDER_TACTICIAN_MODEL")
            ),
            output_content_type=OrderTacticianResponse,
            system_message=(
                """당신은 주문 전술가입니다.
macro_report 및 pulse_report와, 현재 portfolio_ratio를 기반하여 코인의 주문 형태 및 수량을 결정합니다.
매크로 시장 분석 리포트는 일봉 데이터 기준으로 시장의 레짐 종류와 분류 확신도, 자산 노출 비율을 포함하고 있습니다.
펄스 분석 리포트는 15분봉 데이터 기준으로 퍼릇(돌파) 신호가 어떤 형태로 발생했는지와 그 강도를 포함하고 있습니다.

코인의 보유 비율이 0인 경우 short(매도) 주문은 불가능합니다.
또한, macro_report의 exposure의 값만큼만 최대로 매수/매도가 가능합니다.

만일, 기존 보유하고 있는 코인의 비율이 exposure 값보다 높은 경우, 강제 매도 주문을 내야 합니다.
또한, 기존 보유하고 있는 코인의 비율이 exposure의 값과 같은 경우, 펄스가 long이어도, 보유 주문을 내야 합니다.
즉, 투자 비율이 exposure를 넘지 않도록 주문 조정을 조율해야 합니다.

주문 수량은 기존에 보유하고 있던 코인 비율을 제외한, 매매할 비율만 주문 수량으로 설정해야 합니다.
예시로, 기존 코인 보유 비율이 0.4이고, exposure가 0.5이며, long 신호가 발생한 경우, 최대 매수 주문 수량은 0.1이 됩니다.
반대로, 기존 코인 보유 비율이 0.4이고, exposure가 0.3이며, short 신호가 발생한 경우, 최소 매도 주문 수량은 0.1이 됩니다.

### 입력 JSON 형식(예시)
{
    macro_report: {
        regime: bull | bear | sideways | high_volatility,  // 레짐 종류
        confidence: 0‑1,   // 레짐 분류 확신도
        exposure: 0‑1,   // 자산 노출 비율: 전체 자산 중 해당 비율 만큼만 코인에 투자 가능
    },
    pulse_report: {
        pulse: long | short | none, // 돌파 신호 종류
        strength: 0‑1,   // 돌파 신호 강도
    },
    portfolio_ratio: { // 현재 포트폴리오 현황, 각 자산의 비율을 나타냄. cash : 현금, btc: 비트코인(코인 종류)
        "cash": 0.6,
        "btc": 0.4
    },
}

### Output JSON 형식
{order: ..., amount: ...}
                            
### 주문 종류
- long: 매수 포지션
- short: 매도 포지션
- hold: 보유 포지션
                            
### 주문 수량
- (STRICT) 0.0 ~ 1.0 사이 실수값, 소수점 2자리까지 허용
- 0.0: 0% 매수/매도
- 0.5: 50% 매수/매도
- 1.0: 100% 매수/매도
- (STRICT) 보유인 경우 0.0
                            
### 예시
{order: long, amount: 0.5}
{order: short, amount: 0.2}
{order: hold, amount: 0.0}
"""
            ),
        )

    async def decide(
        self,
        macro_report: Dict[str, Any],
        pulse_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        report = {
            "macro_report": macro_report,
            "pulse_report": pulse_report,
            "portfolio_ratio": PortfolioManager.get_instance().get_portfolio_ratio(),
        }
        base_msg = TextMessage(
            content=json.dumps(report, indent=4),
            source="data_preprocessor",
        )
        messages = [base_msg]
        for _ in range(5):  # 최대 5회 반복
            try:
                response = await self.run(task=messages)
                content = response.messages[-1].content
                # pydantic parsing; 범위 벗어나면 error
                OrderResponse.model_validate(content.response.model_dump())

                thoughts = content.thoughts
                order_report = content.response

                self.close()
                return order_report.dict()
            except ValidationError as e:  # ← ValidationError 잡기
                feedback = TextMessage(  # ➌ 스키마 위반 피드백
                    content=(
                        "⛔  JSON schema validation failed:\n"
                        f"{e}\n\n"
                        "규칙:\n"
                        "1. order가 hold인 경우 amount는 0.0.\n"
                        "2. amount는 0.0 ~ 1.0 사이 실수값, 소수점 2자리까지 허용\n"
                    ),
                    source="validator",
                )
            messages.append(feedback)  # ➍ 원본 + 피드백 함께 전달

        raise RuntimeError("OrderTactician: too many invalid responses")
