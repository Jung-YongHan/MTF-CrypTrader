from typing import Any, Dict

from pydantic import BaseModel

from src.agents.macro_team.macro_investment_rate_adjuster import (
    MacroInvestmentRateAdjuster,
    MacroLimitReportResponse,
)
from src.agents.macro_team.regime_analyzer import RegimeAnalyzer, RegimeReportResponse


class MacroReport(BaseModel):
    regime_report: RegimeReportResponse
    macro_limit_report: MacroLimitReportResponse


class MacroAnalysisTeam:
    def __init__(self):
        self.regime_analyzer = RegimeAnalyzer()
        self.macro_investment_rate_adjuster = MacroInvestmentRateAdjuster()

    async def analyze(self, price_data: Dict[str, Any], fig: Any) -> MacroReport:
        """
        Analyze the market regime and adjust the investment rate limit.
        :param price_data: Price data for analysis
        :param fig: Chart figure for analysis
        :return: JSON string of the macro report
        """
        # Analyze the market regime
        regime_report = await self.regime_analyzer.analyze(
            price_data=price_data, fig=fig
        )

        # Adjust the investment rate limit based on the regime report and price_data
        macro_limit_report = (
            await self.macro_investment_rate_adjuster.adjust_rate_limit(
                regime_report=regime_report, price_data=price_data
            )
        )

        # Create the macro report
        macro_report = MacroReport(
            regime_report=regime_report,
            macro_limit_report=macro_limit_report,
        )

        # Convert the macro report to JSON
        return macro_report.model_dump_json()
