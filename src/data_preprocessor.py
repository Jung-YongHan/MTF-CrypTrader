import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import talib

from src.enum.timeframe_category_type import TimeframeCategoryType


class DataPreprocessor:
    """
    - 일봉(상위)과 분봉(하위) 데이터를 구분하여 관리 및 지표 계산
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
            self._compute_macro_indicators()
        if not self.df_micro.empty:
            self._compute_micro_indicators()

    def update_and_get_price_data(
        self, row: dict, timeframe: TimeframeCategoryType, save_path: str = None
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

    def _update(self, row: dict, timeframe: TimeframeCategoryType) -> pd.DataFrame:
        """
        row: dict, 새로운 데이터 한 건
        timeframe: MarketCategoryType
        """
        row_df = pd.DataFrame([row])
        if timeframe == TimeframeCategoryType.HIGHER:
            self.df_macro = (
                pd.concat([self.df_macro, row_df], ignore_index=True)
                .drop_duplicates(subset="datetime", keep="last")
                .sort_values("datetime")
                .reset_index(drop=True)
            )
            self._compute_macro_indicators()
            return self.df_macro
        else:
            self.df_micro = (
                pd.concat([self.df_micro, row_df], ignore_index=True)
                .drop_duplicates(subset="datetime", keep="last")
                .sort_values("datetime")
                .reset_index(drop=True)
            )
            self._compute_micro_indicators()
            return self.df_micro

    def _compute_macro_indicators(self):
        df = self.df_macro
        close, high, low, volume, open_ = (
            df[c].astype(float) for c in ("close", "high", "low", "volume", "open")
        )

        # 거시적(상위) 관점의 주요 지표 계산
        # 경기순환, 추세, 변동성, 위험 등 장기적 흐름 파악에 초점
        df["sma200"] = talib.SMA(close, timeperiod=200)  # 장기 이동평균
        df["ema100"] = talib.EMA(close, timeperiod=100)  # 장기 EMA
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = talib.BBANDS(
            close, timeperiod=60, nbdevup=2, nbdevdn=2
        )  # 장기 볼린저밴드
        df["rsi50"] = talib.RSI(close, timeperiod=50)  # 장기 RSI
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            close, fastperiod=24, slowperiod=52, signalperiod=18
        )  # 장기 MACD
        df["adx50"] = talib.ADX(high, low, close, timeperiod=50)  # 장기 ADX
        df["atr60"] = talib.ATR(high, low, close, timeperiod=60)  # 장기 ATR
        df["stddev60"] = talib.STDDEV(close, timeperiod=60, nbdev=1)  # 장기 표준편차
        df["roc50"] = talib.ROC(close, timeperiod=50)  # 장기 ROC
        df["mom50"] = talib.MOM(close, timeperiod=50)  # 장기 모멘텀
        # 경기순환/위험지표 예시: OBV, MFI 등도 장기 적용
        df["obv"] = talib.OBV(close, volume)
        df["mfi50"] = talib.MFI(high, low, close, volume, timeperiod=50)

    def _compute_micro_indicators(self):
        df = self.df_micro
        close, high, low, _, _ = (
            df[c].astype(float) for c in ("close", "high", "low", "volume", "open")
        )

        # 미시적(하위) 관점의 주요 지표 계산
        # 초단기 모멘텀, 변동성, 시장 미세구조 등 빠른 신호 포착에 초점
        df["sma3"] = talib.SMA(close, timeperiod=3)  # 초단기 이동평균
        df["ema5"] = talib.EMA(close, timeperiod=5)  # 초단기 EMA
        df["rsi7"] = talib.RSI(close, timeperiod=7)  # 초단기 RSI
        df["stoch_k"], df["stoch_d"] = talib.STOCH(
            high, low, close, fastk_period=5, slowk_period=3, slowd_period=3
        )
        df["macd"], df["macd_signal"], df["macd_hist"] = talib.MACD(
            close, fastperiod=6, slowperiod=13, signalperiod=4
        )  # 초단기 MACD
        df["atr5"] = talib.ATR(high, low, close, timeperiod=5)  # 초단기 ATR
        df["stddev5"] = talib.STDDEV(close, timeperiod=5, nbdev=1)  # 초단기 표준편차
        df["roc3"] = talib.ROC(close, timeperiod=3)  # 초단기 ROC
        df["mom3"] = talib.MOM(close, timeperiod=3)  # 초단기 모멘텀
        # 캔들패턴(미시적 신호) 예시
        df["engulfing"] = talib.CDLENGULFING(
            open=df["open"].astype(float), high=high, low=low, close=close
        )
        df["hammer"] = talib.CDLHAMMER(
            open=df["open"].astype(float), high=high, low=low, close=close
        )

    def _draw_close_chart(
        self,
        df: pd.DataFrame,
        timeframe: TimeframeCategoryType = TimeframeCategoryType.HIGHER,
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
