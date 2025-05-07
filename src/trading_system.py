import pandas as pd

from agents.executor import TradeExecutor
from agents.macro.macro_analysis_team import MacroAnalysisTeam
from agents.micro.micro_analysis_team import MicroAnalysisTeam
from agents.reflector import FeedbackReflector
from data_preprocessor import DataPreprocessor


class TradingSystem:
    def __init__(self, regime: str, coin: str):
        self.regime = regime
        self.coin = coin

        def load_data(file_prefix):
            """
            coin: BTC, ETH, SOL
            file_prefix: days, minutes
            regime: bull, bear, total
            """
            return pd.read_csv(f"data/{coin}_{file_prefix}_{regime}.csv")

        self.df_days = load_data("days")
        self.df_minutes = load_data("minutes")

        self.portfolio = {
            "cash": 10_000_000,
            "position": 0,
        }

        self.data_preprocessor = DataPreprocessor()
        self.macro_analysis_team = MacroAnalysisTeam()
        self.micro_analysis_team = MicroAnalysisTeam()
        self.trade_executor = TradeExecutor()
        self.feedback_reflector = FeedbackReflector()

    async def run(self) -> None:
        # 1. 일 단위 데이터를 순회하며 매크로 파라미터를 업데이트
        for index, days in self.df_days.iterrows():
            days_dict = days.to_dict()
            # 2. 현재까지의 일 단위 데이터를 활용, 가격적 분석 지표가 추가된 데이터프레임 생성
            tmp_df = self.data_preprocessor.update(row=days_dict, timeframe="macro")
            # 3. 일 단위 데이터 활용, 차트 그리기
            fig = self.data_preprocessor.draw_close_chart(
                timeframe="macro",
                save_path=f"data/close_charts/{index+1}_day_chart",
                return_fig=True,
            )
            # 4. 레코드 형태로 변환(List[Dict] 형태)
            records = tmp_df.to_dict(orient="records")

            # # 5. 매크로 시장 분석
            regime_report = await self.macro_analysis_team.analyze(records, fig)
            print(f"Regime Report: {regime_report}")
        #     beta = await self.beta_strategist.select(regime)
        #     limit = await self.exposure_limiter.limit(regime)
        #     macro_report = {"regime": regime, "beta": beta, "limit": limit}

        #     # 해당 일의 분봉 데이터만 필터
        #     minutes_slice = self.df_minutes[
        #         self.df_minutes["timestamp"].dt.date == days["timestamp"].date()
        #     ]
        #     for _, minutes in minutes_slice.iterrows():
        #         minutes_dict = minutes.to_dict()
        #         pulse = await self.pulse_detector.detect(minutes_dict, macro_report)
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


if __name__ == "__main__":
    import asyncio
    import warnings

    warnings.filterwarnings("ignore")

    regime = "bull"
    coin = "btc"
    trading_system = TradingSystem(regime, coin)
    asyncio.run(trading_system.run())
