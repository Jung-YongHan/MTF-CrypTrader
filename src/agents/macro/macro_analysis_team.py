from typing import Any, Dict

from src.agents.macro.beta_strategist import BetaStrategist
from src.agents.macro.investment_rate_adjuster import InvestmentRateAdjuster
from src.agents.macro.regime_analyzer import RegimeAnalyzer


class MacroAnalysisTeam:
    def __init__(self):
        self.regime_analyzer = RegimeAnalyzer()
        self.beta_strategist = BetaStrategist()
        self.investment_rate_adjuster = InvestmentRateAdjuster()

    async def analyze(self, price_data: Dict[str, Any], fig: Any) -> str:
        regime_report = await self.regime_analyzer.analyze(
            price_data=price_data, fig=fig
        )

        regime_report = await self.investment_rate_adjuster.adjust_rate_limit(
            regime_report, price_data
        )
        return regime_report
