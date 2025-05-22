from typing import Any, Dict

from src.agents.macro.investment_rate_adjuster import InvestmentRateAdjuster
from src.agents.macro.trend_analyzer import TrendAnalyzer


class MacroAnalysisTeam:
    def __init__(self):
        self.trend_analyzer = TrendAnalyzer()
        self.investment_rate_adjuster = InvestmentRateAdjuster()

    async def analyze(self, price_data: Dict[str, Any], fig: Any) -> str:
        trend_report = await self.trend_analyzer.analyze(price_data=price_data, fig=fig)

        trend_report = await self.investment_rate_adjuster.adjust_rate_limit(
            trend_report, price_data
        )
        return trend_report
