import asyncio
from time import time

import pandas as pd
from dotenv import load_dotenv
from pandas.tseries.offsets import MonthEnd

from src.agents.higher_team.higher_analysis_team import HigherAnalysisTeam
from src.agents.investment_team.investment_expert_team import InvestmentExpertTeam
from src.agents.lower_team.lower_analysis_team import LowerAnalysisTeam
from src.data_preprocessor import DataPreprocessor
from src.enum.market_type import MarketType
from src.enum.record_type import RecordType
from src.enum.tick_type import TickType, get_interval_in_minute
from src.enum.timeframe_category_type import TimeframeCategoryType
from src.enum.trend_type import TrendType
from src.portfoilo_manager import PortfolioManager
from src.record_manager import RecordManager
from src.trade_executor import TradeExecutor


class TradingSystem:
    def __init__(
        self,
        trend: TrendType,
        start_date: str,
        end_date: str,
        market: MarketType,
        higher_tick: TickType,
        lower_tick: TickType,
        only_higher: bool = False,
    ):
        self.trend = trend
        self.start_date = start_date
        self.end_date = end_date
        self.market = market
        self.higher_tick = higher_tick
        self.lower_tick = lower_tick
        self.only_higher = only_higher
        self.initial_balance = 10_000_000

        self.df_higher = self.load_data(market=market, tick=higher_tick)
        self.df_lower = self.load_data(market=market, tick=lower_tick)

        # only_higher가 True면 higher_tick 기준, 아니면 lower_tick 기준
        # 샤프 지수 계산을 위함
        interval_minutes = get_interval_in_minute(
            higher_tick if only_higher else lower_tick
        )
        self.portfolio_manager = PortfolioManager(
            market=market, cash=self.initial_balance, interval_minutes=interval_minutes
        )

        self.data_preprocessor = DataPreprocessor(self.df_higher, self.df_lower)
        self.higher_analysis_team = HigherAnalysisTeam()
        self.lower_analysis_team = LowerAnalysisTeam()
        self.investment_expert_team = InvestmentExpertTeam()
        self.trade_executor = TradeExecutor()

        self.higher_record_manager = RecordManager(
            market=market,
            trend=trend,
            record_type=RecordType.HIGHER,
            only_higher=only_higher,
        )
        self.lower_record_manager = RecordManager(
            market=market, trend=trend, record_type=RecordType.LOWER
        )
        self.trade_record_manager = RecordManager(
            market=market,
            trend=trend,
            record_type=RecordType.TRADE,
            only_higher=only_higher,
        )

    async def run(self) -> None:
        print("Starting backtest...")
        print(f"Market: {self.trend}")
        print(f"Start date: {self.start_date}")
        print(f"End date: {self.end_date}")
        print(f"Coin: {self.market}")
        print(f"Higher tick: {self.higher_tick}")
        print(f"Micro tick: {self.lower_tick}")
        print(f"Only higher: {self.only_higher}")
        print(f"Initial balance: {self.initial_balance}")

        # 1. 상위 단위 데이터를 순회
        start_time = time()
        for index, higher_tick in self.df_higher.iterrows():
            # start_date 이전에 대해서는 가격적 분석 지표만 추가
            higher_price_data = higher_tick.to_dict()

            higher_start_time = time()

            print(f"###### {higher_tick['datetime']} 틱 시작 ######")

            # 2. 현재까지의 상위 단위 데이터를 활용, 가격적 분석 지표 추가 및 차트 생성
            price_data, fig = self.data_preprocessor.update_and_get_price_data(
                row=higher_price_data,
                timeframe=TimeframeCategoryType.HIGHER,
                save_path=f"data/close_charts/{self.trend}/{index+1}_higher_chart",
            )
            # 3. 상위 시장 분석
            higher_report = await self.higher_analysis_team.analyze(
                price_data=price_data, fig=fig
            )

            print(f"Higher Report: {higher_report}")
            higher_report_tmp = dict(higher_report.model_copy())
            higher_report_tmp["datetime"] = higher_tick["datetime"]
            higher_report_tmp["trend"] = higher_report_tmp["trend_report"]["trend"]
            higher_report_tmp["confidence"] = higher_report_tmp["trend_report"][
                "confidence"
            ]
            higher_report_tmp["rate_limit"] = higher_report_tmp["limit_report"][
                "rate_limit"
            ]
            self.higher_record_manager.record_step(higher_report_tmp)

            if abs(higher_report_tmp["limit_report"]["rate_limit"]) < 1e-8:
                print("No rate_limit, skipping lower analysis.")
                continue

            # TODO 거시 미시 독립 및 투자 전문팀 배치에 따른 구조 수정 필요
            if self.only_higher:
                self.portfolio_manager.update_portfolio_ratio(
                    price_data=higher_price_data
                )

                # 4. 상위 시장의 투자 한도에 따라 매매 결정
                trend = higher_report["trend_report"]["trend"]
                rate_limit = higher_report["limit_report"]["rate_limit"]
                coin_ratio = self.portfolio_manager.get_portfolio_ratio()[self.market]

                # 5. 매매 결정
                if trend == TrendType.BULL:
                    if rate_limit > coin_ratio:
                        order = "buy"
                        amount = rate_limit - coin_ratio
                    else:
                        order = "hold"
                        amount = 0.0
                elif trend == TrendType.BEAR:
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
                    price_data=higher_price_data,
                    market=self.market,
                    lower_report={
                        "order_report": {
                            "order": order,
                            "amount": amount,
                        }
                    },
                )

                trade_report = {
                    "datetime": higher_price_data["datetime"],
                    **self.portfolio_manager.get_performance(),
                }
                self.trade_record_manager.record_step(trade_report)

            else:
                # 4. 해당 상위 단위 캔들에 속해있는 하위 데이터만 필터, self.df_lower와 구분됨
                df_lower = self.get_lower_data_for_day(higher_tick=higher_tick)
                # 5. 하위 시장 분석 및 투자 진행
                # 이전 하위 분석 리포트 초기화(시가에 구매를 위해)
                lower_report = None
                for index, lower_tick in df_lower.iterrows():
                    lower_price_data = lower_tick.to_dict()

                    self.portfolio_manager.update_portfolio_ratio(
                        price_data=lower_price_data
                    )

                    # TODO 투자 전문팀 배치, lower_report가 있는 경우에만 매매 결정

                    # 5.1 시가에 대해서 매도/매수/보유 결정
                    await self.trade_executor.execute(
                        price_data=lower_price_data,
                        market=self.market,
                        lower_report=lower_report,
                    )

                    record = {
                        "datetime": lower_price_data["datetime"],
                        **self.portfolio_manager.get_performance(),
                    }
                    self.trade_record_manager.record_step(record)

                    print(f"## {lower_tick['datetime']} 틱 ##")

                    # 6. 현재까지의 하위 단위 데이터를 활용, 가격적 분석 지표 추가 및 차트 생성
                    price_data, fig = self.data_preprocessor.update_and_get_price_data(
                        row=lower_price_data,
                        timeframe=TimeframeCategoryType.LOWER,
                        save_path=f"data/close_charts/{self.trend}/{index+1}_lower_chart",
                    )

                    # 7. 하위 시장 분석 및 주문 결정
                    lower_report = await self.lower_analysis_team.analyze(
                        price_data=price_data,
                        fig=fig,
                    )

                    print(f"Micro Report: {lower_report}")
                    lower_report_tmp = {
                        "datetime": lower_tick["datetime"],
                        "pulse": lower_report["pulse_report"]["pulse"],
                        "strength": lower_report["pulse_report"]["strength"],
                        "order": lower_report["order_report"]["order"],
                        "amount": lower_report["order_report"]["amount"],
                    }
                    self.lower_record_manager.record_step(lower_report_tmp)

                higher_end_time = time()
                print(
                    f"Higher analysis time: {higher_end_time - higher_start_time:.2f} seconds"
                )
        end_time = time()
        print(f"Total time taken for backtest: {end_time - start_time:.2f} seconds")

        await self.portfolio_manager.sell_all(
            price_data=self.df_higher.iloc[-1].to_dict(),
        )

        print("Backtest completed.")
        print(f"Portfolio performance: {self.portfolio_manager.get_performance()}")

    def get_lower_data_for_day(self, higher_tick: pd.Series) -> pd.DataFrame:
        """
        Returns the lower timeframe data (e.g., minute candles) that fall within
        the period specified by the given higher_tick.

        Args:
            higher_tick: A row from the higher DataFrame representing a single
                day or period.

        Returns:
            pd.DataFrame: Micro timeframe data for the specified period.
        """
        day_start = pd.to_datetime(higher_tick["datetime"])

        if self.higher_tick == "month1":
            # Move to the first day of next month, then use it as exclusive upper bound
            day_end = (day_start + MonthEnd(1)) + pd.Timedelta(days=1)
        else:
            day_end = day_start + pd.Timedelta(days=self.higher_tick.days)

        lower_slice = self.df_lower[
            (pd.to_datetime(self.df_lower["datetime"]) >= day_start)
            & (pd.to_datetime(self.df_lower["datetime"]) < day_end)
        ]
        return lower_slice

    def load_data(self, market: MarketType, tick: TickType) -> pd.DataFrame:
        """
        Load data from CSV files based on the specified market and tick type.

        Args:
            market (CoinType): The type of market (e.g., "btc", "eth").
            tick (TickType): The tick type (e.g., "month1", "week1").
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        df = pd.read_csv(f"data/{market}_{tick}.csv")
        df = df[
            (pd.to_datetime(df["datetime"]) >= self.start_date)
            & (pd.to_datetime(df["datetime"]) < self.end_date)
        ]
        return df


class AsyncTradingSystem(TradingSystem):
    def __init__(
        self,
        trend: TrendType,
        start_date: str,
        end_date: str,
        market: MarketType,
        higher_tick: TickType,
        lower_tick: TickType,
        only_higher: bool = False,
    ):
        super().__init__(
            trend=trend,
            start_date=start_date,
            end_date=end_date,
            market=market,
            higher_tick=higher_tick,
            lower_tick=lower_tick,
            only_higher=only_higher,
        )

    def run(self):
        asyncio.run(super().run())


def create_system(
    trend: TrendType,
    start_date: str,
    end_date: str,
    market: MarketType,
    higher_tick: TickType,
    lower_tick: TickType,
    only_higher: bool = False,
):
    import warnings

    warnings.filterwarnings("ignore")

    load_dotenv()

    return AsyncTradingSystem(
        trend=trend,
        start_date=start_date,
        end_date=end_date,
        market=market,
        higher_tick=higher_tick,
        lower_tick=lower_tick,
        only_higher=only_higher,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    trend = "bull"
    market = "btc"
    trading_system = TradingSystem(trend, market)
    asyncio.run(trading_system.run())
