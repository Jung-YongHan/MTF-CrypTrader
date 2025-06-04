from typing import Any, Dict

from pydantic import BaseModel

from src.agents.lower_team.lower_investment_rate_adjuster import (
    LimitReport,
    MicroInvestmentRateAdjuster,
)
from src.agents.lower_team.pulse_detector import PulseDetector, PulseReport


class LowerReport(BaseModel):
    pulse_report: PulseReport
    micro_limit_report: LimitReport


class LowerAnalysisTeam:
    def __init__(self):
        self.pulse_detector = PulseDetector()
        self.micro_investment_rate_adjuster = MicroInvestmentRateAdjuster()

    async def analyze(
        self,
        price_data: Dict[str, Any],
        fig: Any,
    ) -> LowerReport:
        """
        Analyze the market pulse and adjust the investment rate limit.
        :param price_data: Price data for analysis
        :param fig: Chart figure for analysis
        :param macro_report: Higher report for analysis
        :return: JSON string of the micro report
        """
        # Analyze the market pulse
        pulse_report = await self.pulse_detector.detect(price_data=price_data, fig=fig)

        # Adjust the investment rate limit based on the pulse report and price_data
        micro_limit_report = (
            await self.micro_investment_rate_adjuster.adjust_rate_limit(
                pulse_report=pulse_report, price_data=price_data
            )
        )

        # Create the micro report
        micro_report = LowerReport(
            pulse_report=pulse_report,
            micro_limit_report=micro_limit_report,
        )

        # Convert the micro report to JSON
        return micro_report.model_dump_json()
