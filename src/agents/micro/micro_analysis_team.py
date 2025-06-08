from typing import Any, Dict

from src.agents.micro.order_tactician import OrderTactician
from src.agents.micro.pulse_detector import PulseDetector


class MicroAnalysisTeam:
    def __init__(self):
        self.pulse_detector = PulseDetector()
        self.order_tactician = OrderTactician()

    async def analyze(
        self,
        price_data: Dict[str, Any],
        fig: Any,
        macro_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        micro_report = await self.pulse_detector.detect(price_data=price_data, fig=fig)

        order_report = await self.order_tactician.decide(
            macro_report=macro_report, micro_report=micro_report
        )

        micro_report = {"micro_report": micro_report, "order_report": order_report}
        return micro_report
