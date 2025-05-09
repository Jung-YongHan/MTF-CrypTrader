from typing import Any, Dict, List

from src.agents.micro.order_tactician import OrderTactician
from src.agents.micro.pulse_detector import PulseDetector


class MicroAnalysisTeam:
    def __init__(self):
        self.pulse_detector = PulseDetector()
        self.order_tactician = OrderTactician()

    async def analyze(
        self, price_data: List[Dict[str, Any]], fig: Any, portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        pulse_report = await self.pulse_detector.detect(price_data=price_data, fig=fig)
        print(f"Pulse Report: {pulse_report}")

        order = await self.order_tactician.decide(
            pulse_report=pulse_report, portfolio=portfolio
        )
        print(f"Order: {order}")

        return order
