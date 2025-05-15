import asyncio
from time import time

import pandas as pd
from dotenv import load_dotenv
from pandas.tseries.offsets import MonthEnd

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
        macro_tick: str,
        micro_tick: str,
        only_macro: bool = False,
        initial_balance: float = 10_000_000,
    ):
        self.regime = regime
        self.start_date = start_date
        self.end_date = end_date
        self.coin = coin
        self.initial_balance = initial_balance
        self.macro_tick = macro_tick
        self.micro_tick = micro_tick
        self.only_macro = only_macro

        def load_data(tick):
            """
            coin: BTC, ETH, SOL
            tick: __desc__
            """
            if self.end_date:
                return pd.read_csv(f"data/{coin}_{tick}.csv")

        df_macro_full = load_data(macro_tick)
        df_micro_full = load_data(micro_tick)

        self.df_macro = df_macro_full[
            (pd.to_datetime(df_macro_full["datetime"]) >= self.start_date)
            & (pd.to_datetime(df_macro_full["datetime"]) < self.end_date)
        ]
        self.df_micro = df_micro_full[
            (pd.to_datetime(df_micro_full["datetime"]) >= self.start_date)
            & (pd.to_datetime(df_micro_full["datetime"]) < self.end_date)
        ]

        # interval_minutes를 macro_tick, micro_tick에 따라 동적으로 할당
        tick_to_minutes = {
            "month1": 1440 * 30,
            "week1": 1440 * 7,
            "day1": 1440,
            "hour1": 60,
            "minute30": 30,
            "minute15": 15,
            "minute5": 5,
            "minute1": 1,
        }

        # only_macro가 True면 macro_tick 기준, 아니면 micro_tick 기준
        interval_minutes = tick_to_minutes.get(
            macro_tick if only_macro else micro_tick, 1440
        )
        self.portfolio_manager = PortfolioManager(
            coin=coin, cash=initial_balance, interval_minutes=interval_minutes
        )

        self.data_preprocessor = DataPreprocessor(self.df_macro, self.df_micro)
        self.macro_analysis_team = MacroAnalysisTeam()
        self.micro_analysis_team = MicroAnalysisTeam()
        self.trade_executor = TradeExecutor()

        self.macro_recode_manager = RecordManager(
            coin=coin, regime=regime, report_type="macro", only_macro=only_macro
        )
        self.micro_recode_manager = RecordManager(
            coin=coin, regime=regime, report_type="micro"
        )
        self.trade_recode_manager = RecordManager(
            coin=coin, regime=regime, report_type="trade", only_macro=only_macro
        )

    async def run(self) -> None:
        print("Starting backtest...")
        print(f"Regime: {self.regime}")
        print(f"Start date: {self.start_date}")
        print(f"End date: {self.end_date}")
        print(f"Coin: {self.coin}")
        print(f"Macro tick: {self.macro_tick}")
        print(f"Micro tick: {self.micro_tick}")
        print(f"Only macro: {self.only_macro}")
        print(f"Initial balance: {self.initial_balance}")

        # 1. 매크로 단위 데이터를 순회
        start_time = time()
        for index, macro_tick in self.df_macro.iterrows():
            # start_date 이전에 대해서는 가격적 분석 지표만 추가
            macro_dict = macro_tick.to_dict()

            macro_start_time = time()

            print(f"###### {macro_tick['datetime']} 틱 시작 ######")

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
            macro_report_tmp["regime"] = macro_report["regime_report"]["regime"]
            macro_report_tmp["confidence"] = macro_report["regime_report"]["confidence"]
            macro_report_tmp["rate_limit"] = macro_report["limit_report"]["rate_limit"]
            self.macro_recode_manager.record_step(macro_report_tmp)

            if abs(macro_report["limit_report"]["rate_limit"]) < 1e-8:
                print("No rate_limit, skipping micro analysis.")
                continue

            if self.only_macro:
                self.portfolio_manager.update_portfolio_ratio(price_data=macro_dict)

                # 4. 매크로 시장의 투자 한도에 따라 매매 결정
                regime = macro_report["regime_report"]["regime"]
                rate_limit = macro_report["limit_report"]["rate_limit"]
                coin_ratio = self.portfolio_manager.get_portfolio_ratio()[self.coin]

                # 5. 매매 결정
                if regime == "상승장":
                    if rate_limit > coin_ratio:
                        order = "buy"
                        amount = rate_limit - coin_ratio
                    else:
                        order = "hold"
                        amount = 0.0
                elif regime == "하락장":
                    if rate_limit < coin_ratio:
                        order = "sell"
                        amount = coin_ratio - rate_limit
                    else:
                        order = "hold"
                        amount = 0.0
                else:
                    order = "hold"
                    amount = 0.0

                await self.trade_executor.execute(
                    price_data=macro_dict,
                    coin=self.coin,
                    micro_report={
                        "order_report": {
                            "order": order,
                            "amount": amount,
                        }
                    },
                )

                trade_report = {
                    "datetime": macro_dict["datetime"],
                    **self.portfolio_manager.get_performance(),
                }
                self.trade_recode_manager.record_step(trade_report)

            else:
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

                    print(f"## {micro_tick['datetime']} 틱 ##")

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
            price_data=self.df_macro.iloc[-1].to_dict(),
        )

        print("Backtest completed.")
        print(f"Portfolio performance: {self.portfolio_manager.get_performance()}")

    def get_micro_data_for_day(self, macro_tick) -> pd.DataFrame:
        """
        Returns the micro timeframe data (e.g., minute candles) that fall within
        the period specified by the given macro_tick.

        Args:
            macro_tick: A row from the macro DataFrame representing a single
                day or period.

        Returns:
            pd.DataFrame: Micro timeframe data for the specified period.
        """
        days = {
            "week1": 7,
            "day1": 1,
        }

        day_start = pd.to_datetime(macro_tick["datetime"])

        if self.macro_tick == "month1":
            # Move to the first day of next month, then use it as exclusive upper bound
            day_end = (day_start + MonthEnd(1)) + pd.Timedelta(days=1)
        else:
            day_end = day_start + pd.Timedelta(days=days[self.macro_tick])

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
        macro_tick: str,
        micro_tick: str,
        only_macro: bool = False,
    ):
        super().__init__(
            regime=regime,
            start_date=start_date,
            end_date=end_date,
            coin=coin,
            macro_tick=macro_tick,
            micro_tick=micro_tick,
            only_macro=only_macro,
        )

    def run(self):
        asyncio.run(super().run())


def create_system(
    regime: str,
    start_date: str,
    end_date: str,
    coin: str,
    macro_tick: str,
    micro_tick: str,
    only_macro: bool = False,
):
    import warnings

    warnings.filterwarnings("ignore")

    load_dotenv()

    return AsyncTradingSystem(
        regime=regime,
        start_date=start_date,
        end_date=end_date,
        coin=coin,
        macro_tick=macro_tick,
        micro_tick=micro_tick,
        only_macro=only_macro,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    regime = "bull"
    coin = "btc"
    trading_system = TradingSystem(regime, coin)
    asyncio.run(trading_system.run())
