import asyncio

import pandas as pd
from dotenv import load_dotenv

from src.agents.executor import TradeExecutor
from src.agents.macro.macro_analysis_team import MacroAnalysisTeam
from src.agents.micro.micro_analysis_team import MicroAnalysisTeam
from src.agents.reflector import FeedbackReflector
from src.data_preprocessor import DataPreprocessor


class TradingSystem:
    def __init__(self, regime: str, coin: str, micro_tick: int = 1):
        self.regime = regime
        self.coin = coin

        def load_data(file_prefix):
            """
            coin: BTC, ETH, SOL
            file_prefix: day, minute
            regime: bull, bear, total
            """
            if file_prefix == "day":
                return pd.read_csv(f"data/{coin}_{file_prefix}_{regime}.csv")
            elif file_prefix == "minute":
                # micro_tick: 1, 5, 15, 30, 60
                if micro_tick not in [1, 5, 15, 30, 60]:
                    raise ValueError("micro_tick은 1, 3, 5, 15, 30 중 하나여야 합니다.")
                return pd.read_csv(
                    f"data/{coin}_{file_prefix}{micro_tick}_{regime}.csv"
                )
            else:
                raise ValueError("file_prefix는 'day' 또는 'minute'만 가능합니다.")

        self.df_macro = load_data("day")
        self.df_micro = load_data("minute")

        self.portfolio = {
            "cash": 10000000,
            self.coin: 0,
        }

        self.data_preprocessor = DataPreprocessor()
        self.macro_analysis_team = MacroAnalysisTeam()
        self.micro_analysis_team = MicroAnalysisTeam()
        self.trade_executor = TradeExecutor()
        self.feedback_reflector = FeedbackReflector()

    async def run(self) -> None:
        # 1. 일 단위 데이터를 순회하며 매크로 파라미터를 업데이트
        for index, day in self.df_macro.iterrows():
            days_dict = day.to_dict()

            # 2. 현재까지의 일 단위 데이터를 활용, 가격적 분석 지표 추가 및 차트 생성
            records, fig = self.data_preprocessor.update_and_get_data(
                row=days_dict,
                timeframe="macro",
                save_path=f"data/close_charts/{index+1}_day_chart",
            )
            # 3. 매크로 시장 분석
            regime_report = await self.macro_analysis_team.analyze(
                self.portfolio, records, fig
            )
            print(f"Regime Report: {regime_report}")
        #     beta = await self.beta_strategist.select(regime)
        #     limit = await self.exposure_limiter.limit(regime)
        #     macro_report = {"regime": regime, "beta": beta, "limit": limit}

        #     # 해당 일의 분봉 데이터만 필터
        #     minute_slice = self.df_micro[
        #         self.df_micro["timestamp"].dt.date == day["timestamp"].date()
        #     ]
        #     for _, minute in minute_slice.iterrows():
        #         minute_dict = minute.to_dict()
        #         pulse = await self.pulse_detector.detect(minute_dict, macro_report)
        #         order = await self.order_tactician.decide(pulse, self.portfolio)
        #         result = await self.trade_executor.execute(order)
        #         feedback = await self.feedback_reflector.reflect(
        #             {"order": order, "result": result}
        #         )

        #         # 피드백의 patch를 읽어 다음 매크로 파라미터에 적용...
        #         # 예: regime_analyzer.update_system_message(...)
        #         #    beta_strategist.update_system_message(...)
        #         #    exposure_limiter.update_system_message(...)
        #         # 필요에 따라 구현

        # print("Backtest completed.")


class AsyncTradingSystem(TradingSystem):
    def __init__(
        self,
        regime: str,
        coin: str,
        micro_tick: int = 1,
    ):
        super().__init__(
            regime=regime,
            coin=coin,
            micro_tick=micro_tick,
        )

    def run(self):
        asyncio.run(super().run())


def create_system(regime: str, coin: str, micro_tick: int = 1):
    import warnings

    warnings.filterwarnings("ignore")

    load_dotenv()

    return AsyncTradingSystem(
        regime=regime,
        coin=coin,
        micro_tick=micro_tick,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    regime = "bull"
    coin = "btc"
    trading_system = TradingSystem(regime, coin)
    asyncio.run(trading_system.run())
