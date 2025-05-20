from typing import Any, Dict

from pydantic import BaseModel

from src.agents.micro_team.micro_investment_rate_adjuster import (
    MicroInvestmentRateAdjuster,
    MicroLimitReportResponse,
)
from src.agents.micro_team.pulse_detector import PulseDetector, PulseReportResponse


class MicroReport(BaseModel):
    pulse_report: PulseReportResponse
    micro_limit_report: MicroLimitReportResponse


class MicroAnalysisTeam:
    def __init__(self):
        self.pulse_detector = PulseDetector()
        self.micro_investment_rate_adjuster = MicroInvestmentRateAdjuster()

    async def analyze(
        self,
        price_data: Dict[str, Any],
        fig: Any,
    ) -> MicroReport:
        """
        Analyze the market pulse and adjust the investment rate limit.
        :param price_data: Price data for analysis
        :param fig: Chart figure for analysis
        :param macro_report: Macro report for analysis
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
        micro_report = MicroReport(
            pulse_report=pulse_report,
            micro_limit_report=micro_limit_report,
        )

        # Convert the micro report to JSON
        return micro_report.model_dump_json()
