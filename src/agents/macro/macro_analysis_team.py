from typing import Any, Dict

from src.agents.macro.beta_strategist import BetaStrategist
from src.agents.macro.exposure_limiter import ExposureLimiter
from src.agents.macro.regime_analyzer import RegimeAnalyzer


class MacroAnalysisTeam:
    def __init__(self):
        self.regime_analyzer = RegimeAnalyzer()
        self.beta_strategist = BetaStrategist()
        self.exposure_limiter = ExposureLimiter()

    async def analyze(
        self, portfolio: Dict[str, Any], price_data: Dict[str, Any], fig: Any
    ) -> str:
        regime_report = await self.regime_analyzer.analyze(
            price_data=price_data, fig=fig
        )

        # # 베타 전략가에게 레짐 전달
        # beta = await self.beta_strategist.select(regime_report.regime)
        # print(f"Beta: {beta}")

        # 익스포저 리미터에게 레짐 전달 후 익스포저 리미터가 계산된 리포트 반환
        regime_report = await self.exposure_limiter.limit_exposure(
            portfolio, regime_report, price_data
        )
        return regime_report
