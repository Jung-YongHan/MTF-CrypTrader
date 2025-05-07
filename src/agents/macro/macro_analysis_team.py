from typing import Any, Dict, List

from agents.macro.beta_strategist import BetaStrategist
from agents.macro.exposure_limiter import ExposureLimiter
from agents.macro.regime_analyzer import RegimeAnalyzer


class MacroAnalysisTeam:
    def __init__(self):
        self.regime_analyzer = RegimeAnalyzer()
        self.beta_strategist = BetaStrategist()
        self.exposure_limiter = ExposureLimiter()

    async def analyze(self, price_data: List[Dict[str, Any]], fig: Any) -> str:
        regime_report = await self.regime_analyzer.analyze(
            price_data=price_data, fig=fig
        )

        # # 베타 전략가에게 레짐 전달
        # beta = await self.beta_strategist.select(regime_report.regime)
        # print(f"Beta: {beta}")

        # # 익스포저 리미터에게 레짐 전달
        # limit = await self.exposure_limiter.limit(regime_report.regime)
        # print(f"Limit: {limit}")

        return regime_report
