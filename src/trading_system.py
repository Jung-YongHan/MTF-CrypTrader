import asyncio
from time import time

import pandas as pd
from dotenv import load_dotenv
from pandas.tseries.offsets import MonthEnd

from src.agents.investment_team.investment_expert_team import InvestmentExpertTeam
from src.agents.macro_team.macro_analysis_team import MacroAnalysisTeam
from src.agents.micro_team.micro_analysis_team import MicroAnalysisTeam
from src.data_preprocessor import DataPreprocessor
from src.enum.coin_type import CoinType
from src.enum.market_category_type import MarketCategoryType
from src.enum.market_type import MarketType
from src.enum.record_type import RecordType
from src.enum.regime_type import RegimeType
from src.enum.tick_type import TickType, get_interval_in_minute
from src.portfoilo_manager import PortfolioManager
from src.record_manager import RecordManager
from src.trade_executor import TradeExecutor


class TradingSystem:
    def __init__(
        self,
        market: MarketType,
        start_date: str,
        end_date: str,
        coin: CoinType,
        macro_tick: TickType,
        micro_tick: TickType,
        only_macro: bool = False,
    ):
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.coin = coin
        self.macro_tick = macro_tick
        self.micro_tick = micro_tick
        self.only_macro = only_macro
        self.initial_balance = 10_000_000

        self.df_macro = self.load_data(coin=coin, tick=macro_tick)
        self.df_micro = self.load_data(coin=coin, tick=micro_tick)

        # only_macro가 True면 macro_tick 기준, 아니면 micro_tick 기준
        # 샤프 지수 계산을 위함
        interval_minutes = get_interval_in_minute(
            macro_tick if only_macro else micro_tick
        )
        self.portfolio_manager = PortfolioManager(
            coin=coin, cash=self.initial_balance, interval_minutes=interval_minutes
        )

        self.data_preprocessor = DataPreprocessor(self.df_macro, self.df_micro)
        self.macro_analysis_team = MacroAnalysisTeam()
        self.micro_analysis_team = MicroAnalysisTeam()
        self.investment_expert_team = InvestmentExpertTeam()
        self.trade_executor = TradeExecutor()

        self.macro_record_manager = RecordManager(
            coin=coin,
            market=market,
            record_type=RecordType.MACRO,
            only_macro=only_macro,
        )
        self.micro_record_manager = RecordManager(
            coin=coin, market=market, record_type=RecordType.MICRO
        )
        self.trade_record_manager = RecordManager(
            coin=coin,
            market=market,
            record_type=RecordType.TRADE,
            only_macro=only_macro,
        )

    async def run(self) -> None:
        print("Starting backtest...")
        print(f"Market: {self.market}")
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
            macro_price_data = macro_tick.to_dict()

            macro_start_time = time()

            print(f"###### {macro_tick['datetime']} 틱 시작 ######")

            # 2. 현재까지의 매크로 단위 데이터를 활용, 가격적 분석 지표 추가 및 차트 생성
            price_data, fig = self.data_preprocessor.update_and_get_price_data(
                row=macro_price_data,
                timeframe=MarketCategoryType.MACRO,
                save_path=f"data/close_charts/{self.market}/{index+1}_macro_chart",
            )
            # 3. 매크로 시장 분석
            macro_report = await self.macro_analysis_team.analyze(
                price_data=price_data, fig=fig
            )

            print(f"Macro Report: {macro_report}")
            macro_report_tmp = macro_report.model_copy()
            macro_report_tmp["datetime"] = macro_tick["datetime"]
            macro_report_tmp["market"] = macro_report["market_report"]["market"]
            macro_report_tmp["confidence"] = macro_report["market_report"]["confidence"]
            macro_report_tmp["rate_limit"] = macro_report["limit_report"]["rate_limit"]
            self.macro_record_manager.record_step(macro_report_tmp)

            if abs(macro_report["limit_report"]["rate_limit"]) < 1e-8:
                print("No rate_limit, skipping micro analysis.")
                continue

            # TODO 거시 미시 독립 및 투자 전문팀 배치에 따른 구조 수정 필요
            if self.only_macro:
                self.portfolio_manager.update_portfolio_ratio(
                    price_data=macro_price_data
                )

                # 4. 매크로 시장의 투자 한도에 따라 매매 결정
                market = macro_report["market_report"]["market"]
                rate_limit = macro_report["limit_report"]["rate_limit"]
                coin_ratio = self.portfolio_manager.get_portfolio_ratio()[self.coin]

                # 5. 매매 결정
                if market == RegimeType.BULL:
                    if rate_limit > coin_ratio:
                        order = "buy"
                        amount = rate_limit - coin_ratio
                    else:
                        order = "hold"
                        amount = 0.0
                elif market == RegimeType.BEAR:
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
                    price_data=macro_price_data,
                    coin=self.coin,
                    micro_report={
                        "order_report": {
                            "order": order,
                            "amount": amount,
                        }
                    },
                )

                trade_report = {
                    "datetime": macro_price_data["datetime"],
                    **self.portfolio_manager.get_performance(),
                }
                self.trade_record_manager.record_step(trade_report)

            else:
                # 4. 해당 매크로 단위 캔들에 속해있는 마이크로 데이터만 필터, self.df_micro와 구분됨
                df_micro = self.get_micro_data_for_day(macro_tick=macro_tick)
                # 5. 마이크로 시장 분석 및 투자 진행
                # 이전 마이크로 분석 리포트 초기화(시가에 구매를 위해)
                micro_report = None
                for index, micro_tick in df_micro.iterrows():
                    micro_price_data = micro_tick.to_dict()

                    self.portfolio_manager.update_portfolio_ratio(
                        price_data=micro_price_data
                    )

                    # TODO 투자 전문팀 배치, micro_report가 있는 경우에만 매매 결정

                    # 5.1 시가에 대해서 매도/매수/보유 결정
                    await self.trade_executor.execute(
                        price_data=micro_price_data,
                        coin=self.coin,
                        micro_report=micro_report,
                    )

                    record = {
                        "datetime": micro_price_data["datetime"],
                        **self.portfolio_manager.get_performance(),
                    }
                    self.trade_record_manager.record_step(record)

                    print(f"## {micro_tick['datetime']} 틱 ##")

                    # 6. 현재까지의 마이크로 단위 데이터를 활용, 가격적 분석 지표 추가 및 차트 생성
                    price_data, fig = self.data_preprocessor.update_and_get_price_data(
                        row=micro_price_data,
                        timeframe=MarketCategoryType.MICRO,
                        save_path=f"data/close_charts/{self.market}/{index+1}_micro_chart",
                    )

                    # 7. 마이크로 시장 분석 및 주문 결정
                    micro_report = await self.micro_analysis_team.analyze(
                        price_data=price_data,
                        fig=fig,
                    )

                    print(f"Micro Report: {micro_report}")
                    micro_report_tmp = {
                        "datetime": micro_tick["datetime"],
                        "pulse": micro_report["pulse_report"]["pulse"],
                        "strength": micro_report["pulse_report"]["strength"],
                        "order": micro_report["order_report"]["order"],
                        "amount": micro_report["order_report"]["amount"],
                    }
                    self.micro_record_manager.record_step(micro_report_tmp)

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

    def get_micro_data_for_day(self, macro_tick: pd.Series) -> pd.DataFrame:
        """
        Returns the micro timeframe data (e.g., minute candles) that fall within
        the period specified by the given macro_tick.

        Args:
            macro_tick: A row from the macro DataFrame representing a single
                day or period.

        Returns:
            pd.DataFrame: Micro timeframe data for the specified period.
        """
        day_start = pd.to_datetime(macro_tick["datetime"])

        if self.macro_tick == "month1":
            # Move to the first day of next month, then use it as exclusive upper bound
            day_end = (day_start + MonthEnd(1)) + pd.Timedelta(days=1)
        else:
            day_end = day_start + pd.Timedelta(days=self.macro_tick.days)

        micro_slice = self.df_micro[
            (pd.to_datetime(self.df_micro["datetime"]) >= day_start)
            & (pd.to_datetime(self.df_micro["datetime"]) < day_end)
        ]
        return micro_slice

    def load_data(self, coin: CoinType, tick: TickType) -> pd.DataFrame:
        """
        Load data from CSV files based on the specified coin and tick type.

        Args:
            coin (CoinType): The type of coin (e.g., "btc", "eth").
            tick (TickType): The tick type (e.g., "month1", "week1").
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        df = pd.read_csv(f"data/{coin}_{tick}.csv")
        df = df[
            (pd.to_datetime(df["datetime"]) >= self.start_date)
            & (pd.to_datetime(df["datetime"]) < self.end_date)
        ]
        return df


class AsyncTradingSystem(TradingSystem):
    def __init__(
        self,
        market: MarketType,
        start_date: str,
        end_date: str,
        coin: CoinType,
        macro_tick: TickType,
        micro_tick: TickType,
        only_macro: bool = False,
    ):
        super().__init__(
            market=market,
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
    market: MarketType,
    start_date: str,
    end_date: str,
    coin: CoinType,
    macro_tick: TickType,
    micro_tick: TickType,
    only_macro: bool = False,
):
    import warnings

    warnings.filterwarnings("ignore")

    load_dotenv()

    return AsyncTradingSystem(
        market=market,
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

    market = "bull"
    coin = "btc"
    trading_system = TradingSystem(market, coin)
    asyncio.run(trading_system.run())
