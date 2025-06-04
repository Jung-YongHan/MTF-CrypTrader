from typing import Any, Dict

from pydantic import BaseModel

from src.agents.higher_team.higher_investment_rate_adjuster import (
    HigherInvestmentRateAdjuster,
    LimitReport,
)
from src.agents.higher_team.trend_analyzer import TrendAnalyzer, TrendReport


class HigherReport(BaseModel):
    trend_report: TrendReport
    limit_report: LimitReport


class HigherAnalysisTeam:
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.higher_investment_rate_adjuster = HigherInvestmentRateAdjuster()

    async def analyze(self, price_data: Dict[str, Any], fig: Any) -> HigherReport:
        """
        Analyze the market trend and adjust the investment rate limit.
        :param price_data: Price data for analysis
        :param fig: Chart figure for analysis
        :return: JSON string of the higher report
        """
        # Analyze the market trend
        trend_report = await self.trend_analyzer.analyze(price_data=price_data, fig=fig)

        # Adjust the investment rate limit based on the trend report and price_data
        limit_report = await self.higher_investment_rate_adjuster.adjust_rate_limit(
            trend_report=trend_report, price_data=price_data
        )

        print(f"Higher Analysis Team: {trend_report}, {limit_report}")
        print(f"Higher Analysis Team: {type(trend_report)}, {type(limit_report)}")

        # Create the higher report
        higher_report = HigherReport(
            trend_report=trend_report,
            limit_report=limit_report,
        )

        return higher_report
