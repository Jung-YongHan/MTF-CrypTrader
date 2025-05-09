import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import talib


class DataPreprocessor:
    """
    - 일봉(매크로)과 분봉(마이크로) 데이터를 구분하여 관리 및 지표 계산
    - update()로 새로운 데이터(딕셔너리) 한 건씩 받아
      1) 내부 DataFrame(일봉/분봉)에 append
      2) 각 시장에 맞는 주요 지표 재계산
    """

    def __init__(self):
        # 일봉(매크로) 데이터프레임
        self.df_macro = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        # 분봉(마이크로) 데이터프레임
        self.df_micro = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )

    def update_and_get_data(
        self, row: dict, timeframe: str, save_path: str
    ) -> Tuple[Dict, Any]:
        full_df = self._update(row, timeframe)

        if timeframe == "macro":
            window_df = full_df.tail(60)
        else:
            window_df = full_df.tail(20)

        fig = self._draw_close_chart(
            df=window_df, timeframe=timeframe, save_path=save_path, return_fig=True
        )
        tmp_df = full_df.iloc[[-1]].dropna(axis=1).to_dict(orient="records")[0]
        return tmp_df, fig

    def _update(self, row: dict, timeframe: str) -> pd.DataFrame:
        """
        row: dict, 새로운 데이터 한 건
        timeframe: "macro" (일봉) 또는 "micro" (분봉)
        """
        row_df = pd.DataFrame([row])
        if timeframe == "macro":
            self.df_macro = pd.concat([self.df_macro, row_df], ignore_index=True)
            self._compute_macro_indicators()
            return self.df_macro
        elif timeframe == "micro":
            self.df_micro = pd.concat([self.df_micro, row_df], ignore_index=True)
            self._compute_micro_indicators()
            return self.df_micro
        else:
            raise ValueError("timeframe은 'macro' 또는 'micro'만 가능합니다.")

    def _compute_macro_indicators(self):
        df = self.df_macro
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        open_ = df["open"].astype(float)

        # 일봉용 주요 지표 계산 (기존과 동일)
        df["sma5"] = talib.SMA(close, timeperiod=5)
        df["sma20"] = talib.SMA(close, timeperiod=20)
        df["sma60"] = talib.SMA(close, timeperiod=60)
        df["ema12"] = talib.EMA(close, timeperiod=12)
        df["ema26"] = talib.EMA(close, timeperiod=26)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df["rsi14"] = talib.RSI(close, timeperiod=14)
        df["stoch_k"], df["stoch_d"] = talib.STOCH(high, low, close)
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        df["cci"] = talib.CCI(high, low, close, timeperiod=20)
        df["adx"] = talib.ADX(high, low, close, timeperiod=14)
        df["obv"] = talib.OBV(close, volume)
        df["mfi"] = talib.MFI(high, low, close, volume, timeperiod=14)
        df["atr"] = talib.ATR(high, low, close, timeperiod=14)
        df["trange"] = talib.TRANGE(high, low, close)
        df["stddev20"] = talib.STDDEV(close, timeperiod=20, nbdev=1)
        df["roc"] = talib.ROC(close, timeperiod=10)
        df["mom"] = talib.MOM(close, timeperiod=10)
        df["cdl_engulfing"] = talib.CDLENGULFING(open_, high, low, close)
        df["cdl_hammer"] = talib.CDLHAMMER(open_, high, low, close)

    def _compute_micro_indicators(self):
        df = self.df_micro
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        open_ = df["open"].astype(float)

        # 분봉용 지표: 일부 중복, 일부 간소화/특화 가능
        df["sma5"] = talib.SMA(close, timeperiod=5)
        df["sma20"] = talib.SMA(close, timeperiod=20)
        df["ema12"] = talib.EMA(close, timeperiod=12)
        df["ema26"] = talib.EMA(close, timeperiod=26)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        df["rsi14"] = talib.RSI(close, timeperiod=14)
        df["stoch_k"], df["stoch_d"] = talib.STOCH(high, low, close)
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        # 마이크로 시장 특화: ATR, 표준편차, 모멘텀 등 빠른 변동성 지표 위주
        df["atr"] = talib.ATR(high, low, close, timeperiod=14)
        df["stddev20"] = talib.STDDEV(close, timeperiod=20, nbdev=1)
        df["roc"] = talib.ROC(close, timeperiod=5)
        df["mom"] = talib.MOM(close, timeperiod=5)
        # 캔들패턴은 필요시 추가

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

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["datetime"], df["close"], label="Close")
        ax.set_title(f"Close Price Over Time ({timeframe})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.autofmt_xdate(rotation=45)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10, prune="both"))

        if save_path:
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
