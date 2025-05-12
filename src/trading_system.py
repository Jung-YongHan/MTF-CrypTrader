import asyncio
from time import time

import pandas as pd
from dotenv import load_dotenv

from src.agents.macro.macro_analysis_team import MacroAnalysisTeam
from src.agents.micro.micro_analysis_team import MicroAnalysisTeam
from src.data_preprocessor import DataPreprocessor
from src.portfoilo_manager import PortfolioManager
from src.record_manager import RecordManager
from src.trade_executor import TradeExecutor


class TradingSystem:
    def __init__(
        self,
        regime: str,
        start_date: str,
        end_date: str,
        coin: str,
        micro_tick: int = 1,
        initial_balance: float = 10_000_000,
    ):
        self.regime = regime
        self.start_date = start_date
        self.end_date = end_date
        self.coin = coin
        self.initial_balance = initial_balance

        def load_data(file_prefix):
            """
            coin: BTC, ETH, SOL
            file_prefix: day, minute
            regime: bull, bear, total
            """
            if file_prefix == "day":
                # end_date 까지만 분석
                if self.end_date:
                    return pd.read_csv(f"data/{coin}_{file_prefix}.csv")
            elif file_prefix == "minute":
                # micro_tick: 1, 3, 5, 15, 30
                if micro_tick not in [1, 3, 5, 15, 30]:
                    raise ValueError("micro_tick은 1, 3, 5, 15, 30 중 하나여야 합니다.")
                # end_date 까지만 분석
                if self.end_date:
                    return pd.read_csv(f"data/{coin}_{file_prefix}{micro_tick}.csv")
            else:
                raise ValueError("file_prefix는 'day' 또는 'minute'만 가능합니다.")

        self.df_macro = load_data("day")
        self.df_micro = load_data("minute")

        self.portfolio_manager = PortfolioManager(
            coin=coin,
            cash=initial_balance,
        )

        self.data_preprocessor = DataPreprocessor(self.df_macro, self.df_micro)
        self.macro_analysis_team = MacroAnalysisTeam()
        self.micro_analysis_team = MicroAnalysisTeam()
        self.trade_executor = TradeExecutor()

        self.macro_recode_manager = RecordManager(
            coin=coin, regime=regime, report_type="macro"
        )
        self.micro_recode_manager = RecordManager(
            coin=coin, regime=regime, report_type="micro"
        )
        self.trade_recode_manager = RecordManager(
            coin=coin, regime=regime, report_type="trade"
        )

        self.df_macro = self.df_macro[
            (self.df_macro["datetime"] >= self.start_date)
            & (self.df_macro["datetime"] < self.end_date)
        ]
        self.df_micro = self.df_micro[
            (self.df_micro["datetime"] >= self.start_date)
            & (self.df_micro["datetime"] < self.end_date)
        ]

    async def run(self) -> None:
        # 1. 매크로 단위 데이터를 순회
        start_time = time()
        for index, macro_tick in self.df_macro.iterrows():
            # start_date 이전에 대해서는 가격적 분석 지표만 추가
            macro_dict = macro_tick.to_dict()

            macro_start_time = time()

            print(f"###### {macro_tick['datetime']}일 ######")

            # 2. 현재까지의 매크로 단위 데이터를 활용, 가격적 분석 지표 추가 및 차트 생성
            price_data, fig = self.data_preprocessor.update_and_get_price_data(
                row=macro_dict,
                timeframe="macro",
                save_path=f"data/close_charts/{self.regime}/{index+1}_macro_chart",
            )
            # 3. 매크로 시장 분석
            macro_report = await self.macro_analysis_team.analyze(
                price_data=price_data, fig=fig
            )

            print(f"Macro Report: {macro_report}")
            macro_report_tmp = macro_report.copy()
            macro_report_tmp["datetime"] = macro_tick["datetime"]
            self.macro_recode_manager.record_step(macro_report_tmp)

            if abs(macro_report["rate_limit"]) < 1e-8:
                print("No rate_limit, skipping micro analysis.")
                continue

            # 4. 해당 매크로 단위 캔들에 속해있는 마이크로 데이터만 필터, self.df_micro와 구분됨
            df_micro = self.get_micro_data_for_day(macro_tick=macro_tick)

            # 5. 마이크로 시장 분석 및 투자 진행
            # 이전 마이크로 분석 리포트 초기화(시가에 구매를 위해)
            micro_report = None
            for index, micro_tick in df_micro.iterrows():
                micro_dict = micro_tick.to_dict()

                self.portfolio_manager.update_portfolio_ratio(price_data=micro_dict)

                # 5.1 시가에 대해서 매도/매수/보유 결정
                await self.trade_executor.execute(
                    price_data=micro_dict,
                    coin=self.coin,
                    micro_report=micro_report,
                )

                trade_report = {
                    "datetime": micro_dict["datetime"],
                    **self.portfolio_manager.get_performance(),
                }
                self.trade_recode_manager.record_step(trade_report)

                print(f"## {micro_tick['datetime']} 캔들 ##")

                # 6. 현재까지의 마이크로 단위 데이터를 활용, 가격적 분석 지표 추가 및 차트 생성
                price_data, fig = self.data_preprocessor.update_and_get_price_data(
                    row=micro_dict,
                    timeframe="micro",
                    save_path=f"data/close_charts/{self.regime}/{index+1}_micro_chart",
                )

                # 7. 마이크로 시장 분석 및 주문 결정
                micro_report = await self.micro_analysis_team.analyze(
                    price_data=price_data,
                    fig=fig,
                    macro_report=macro_report,
                )

                print(f"Micro Report: {micro_report}")
                micro_report_tmp = {
                    "datetime": micro_tick["datetime"],
                    "pulse": micro_report["pulse_report"]["pulse"],
                    "strength": micro_report["pulse_report"]["strength"],
                    "order": micro_report["order_report"]["order"],
                    "amount": micro_report["order_report"]["amount"],
                }
                self.micro_recode_manager.record_step(micro_report_tmp)

            macro_end_time = time()
            print(
                f"Macro analysis time: {macro_end_time - macro_start_time:.2f} seconds"
            )
        end_time = time()
        print(f"Total time taken for backtest: {end_time - start_time:.2f} seconds")

        await self.portfolio_manager.sell_all(
            price=self.df_macro.iloc[-1]["close"],
        )

        print("Backtest completed.")
        print(f"Portfolio performance: {self.portfolio_manager.get_performance()}")

    def get_micro_data_for_day(self, macro_tick) -> pd.DataFrame:
        """
        Returns the micro timeframe data (e.g., minute candles) that fall within the day specified by the given macro_tick.

        Args:
            macro_tick: A row from the macro DataFrame representing a single day.

        Returns:
            pd.DataFrame: Micro timeframe data for the specified day.
        """
        day_start = pd.to_datetime(macro_tick["datetime"])
        day_end = day_start + pd.Timedelta(days=1)
        micro_slice = self.df_micro[
            (pd.to_datetime(self.df_micro["datetime"]) >= day_start)
            & (pd.to_datetime(self.df_micro["datetime"]) < day_end)
        ]
        return micro_slice


class AsyncTradingSystem(TradingSystem):
    def __init__(
        self,
        regime: str,
        start_date: str,
        end_date: str,
        coin: str,
        micro_tick: int = 1,
    ):
        super().__init__(
            regime=regime,
            start_date=start_date,
            end_date=end_date,
            coin=coin,
            micro_tick=micro_tick,
        )

    def run(self):
        asyncio.run(super().run())


def create_system(
    regime: str, start_date: str, end_date: str, coin: str, micro_tick: int = 1
):
    import warnings

    warnings.filterwarnings("ignore")

    load_dotenv()

    return AsyncTradingSystem(
        regime=regime,
        start_date=start_date,
        end_date=end_date,
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
