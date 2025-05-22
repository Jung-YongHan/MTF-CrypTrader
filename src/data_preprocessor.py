import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import talib


class DataPreprocessor:
    """
    - 일봉(매크로)과 분봉(마이크로) 데이터를 구분하여 관리 및 지표 계산
    - update()로 새로운 데이터(딕셔너리) 한 건씩 받아
      1) 내부 DataFrame(일봉/분봉)에 append
      2) 각 시장에 맞는 주요 지표 재계산
    """

    def __init__(
        self, df_macro: pd.DataFrame | None = None, df_micro: pd.DataFrame | None = None
    ):
        # 기본 컬럼 정의
        base_cols = ["datetime", "open", "high", "low", "close", "volume"]

        # 주입된 초기 데이터프레임이 없으면 빈 DF 생성
        self.df_macro = (
            df_macro.copy() if df_macro is not None else pd.DataFrame(columns=base_cols)
        )
        self.df_micro = (
            df_micro.copy() if df_micro is not None else pd.DataFrame(columns=base_cols)
        )

        # 공통 전처리: dtype·정렬·중복 제거
        for df in (self.df_macro, self.df_micro):
            if not df.empty:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.drop_duplicates(subset="datetime", keep="last", inplace=True)
                df.sort_values("datetime", inplace=True)
                df.reset_index(drop=True, inplace=True)

        # 초기 지표 계산
        if not self.df_macro.empty:
            self._compute_higher_timeframe_indicators()
        if not self.df_micro.empty:
            self._compute_lower_timeframe_indicators()

    def update_and_get_price_data(
        self, row: dict, timeframe: str, save_path: str = None
    ) -> Tuple[Dict, Any]:
        # datetime 파싱
        row["datetime"] = pd.to_datetime(row["datetime"])

        full_df = self._update(row, timeframe)
        full_df["datetime"] = pd.to_datetime(full_df["datetime"])

        # 기간(window) 설정
        window = 40

        # row 시점까지의 과거 데이터 확보
        hist_df = full_df[full_df["datetime"] <= row["datetime"]]
        window_df = hist_df.tail(window)  # 부족하면 가용 범위 전체

        fig = self._draw_close_chart(
            df=window_df, timeframe=timeframe, save_path=save_path, return_fig=True
        )
        # row 시점(가장 최근 행)만 dict 로 변환해 반환
        latest_row = hist_df.iloc[[-1]].dropna(axis=1).to_dict(orient="records")[0]
        latest_row["datetime"] = latest_row["datetime"].strftime("%Y-%m-%d %H:%M:%S")
        return latest_row, fig  # tmp_df를 latest_row로 변경하여 반환

    def _update(self, row: dict, timeframe: str) -> pd.DataFrame:
        """
        row: dict, 새로운 데이터 한 건
        timeframe: "macro" 또는 "micro"
        """
        row_df = pd.DataFrame([row])
        if timeframe == "macro":
            self.df_macro = (
                pd.concat([self.df_macro, row_df], ignore_index=True)
                .drop_duplicates(subset="datetime", keep="last")
                .sort_values("datetime")
                .reset_index(drop=True)
            )
            self._compute_higher_timeframe_indicators()
            return self.df_macro
        elif timeframe == "micro":
            self.df_micro = (
                pd.concat([self.df_micro, row_df], ignore_index=True)
                .drop_duplicates(subset="datetime", keep="last")
                .sort_values("datetime")
                .reset_index(drop=True)
            )
            self._compute_lower_timeframe_indicators()
            return self.df_micro
        else:
            raise ValueError("timeframe은 'macro' 또는 'micro'만 가능합니다.")

    def _compute_higher_timeframe_indicators(self):
        df = self.df_macro
        close, high, low, volume, open_ = (
            df[c].astype(float) for c in ("close", "high", "low", "volume", "open")
        )

        df["sma5"] = talib.SMA(close, timeperiod=5)
        df["sma10"] = talib.SMA(close, timeperiod=10)
        df["sma20"] = talib.SMA(close, timeperiod=20)
        df["ema5"] = talib.EMA(close, timeperiod=5)
        df["ema10"] = talib.EMA(close, timeperiod=10)
        df["ema20"] = talib.EMA(close, timeperiod=20)
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["sar"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    def _compute_lower_timeframe_indicators(self):
        df = self.df_micro
        close, high, low, _, _ = (
            df[c].astype(float) for c in ("close", "high", "low", "volume", "open")
        )
        # 참고: https://realtrading.com/trading-blog/short-term-trading-indicators/

        df["sma5"] = talib.SMA(close, timeperiod=5)
        df["sma10"] = talib.SMA(close, timeperiod=10)
        df["sma20"] = talib.SMA(close, timeperiod=20)
        df["ema5"] = talib.EMA(close, timeperiod=5)
        df["ema10"] = talib.EMA(close, timeperiod=10)
        df["ema20"] = talib.EMA(close, timeperiod=20)
        df["rsi"] = talib.RSI(close, timeperiod=14)
        df["stoch_k"], df["stoch_d"] = talib.STOCHF(
            high, low, close, fastk_period=14, fastd_period=3, fastd_matype=0
        )
        df["adx"] = talib.ADX(high, low, close, timeperiod=14)
        upperband, middleband, lowerband = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df["bb_upper"] = upperband
        df["bb_middle"] = middleband
        df["bb_lower"] = lowerband

    def _draw_close_chart(
        self,
        df: pd.DataFrame,
        timeframe: str = "macro",
        save_path: str = None,
        return_fig: bool = False,
    ):
        """
        현재까지 누적된 종가 시계열을 선 그래프로 표시
        - save_path: 파일로 저장할 경로(str), None이면 저장하지 않음
        - return_fig: True면 Figure 객체 반환 (멀티모달 에이전트 전달용)
        """
        if df.empty or df["close"].isnull().all():
            if return_fig:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                return fig
            return

        # mplfinance expects columns: Open, High, Low, Close (case-insensitive)
        # Ensure columns are properly named and datetime is index
        plot_df = df.copy()
        plot_df = plot_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        plot_df.set_index("datetime", inplace=True)

        fig, axlist = mpf.plot(
            plot_df,
            type="candle",
            style="charles",
            title="Candlestick Chart (Price Only)",
            ylabel="close",
            volume=False,
            returnfig=True,
            figsize=(10, 4),
        )
        fig.tight_layout()

        if save_path is not None:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            try:
                fig.savefig(save_path)
            except Exception as e:
                print(f"Error saving chart to {save_path}: {e}")

        if return_fig:
            return fig
        plt.close(fig)
