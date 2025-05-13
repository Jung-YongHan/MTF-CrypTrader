import os
from typing import Any, Dict

import pandas as pd


class RecordManager:
    def __init__(
        self, coin: str, regime: str, report_type: str, only_macro: bool = False
    ):
        folder_path = "only_macro" if only_macro else "results"

        self.folder_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                f"../data/{folder_path}/{regime}/{report_type}",
            )
        )
        os.makedirs(self.folder_path, exist_ok=True)

        self.file_path = os.path.join(self.folder_path, f"{coin}_{regime}.csv")

        if report_type == "macro":
            self.column_types = {
                "datetime": "datetime64[ns]",
                "regime": "object",
                "confidence": "float64",
                "rate_limit": "float64",
            }
        elif report_type == "micro":
            self.column_types = {
                "datetime": "datetime64[ns]",
                "pulse": "object",
                "strength": "float64",
                "order": "object",
                "amount": "float64",
            }
        elif report_type == "trade":
            self.column_types = {
                "datetime": "datetime64[ns]",
                "return": "float64",
                "mdd": "float64",
                "sharpe": "float64",
            }
        else:
            raise ValueError(f"Unknown report_type: {report_type}")

        # 👉 이미 파일이 존재하면 지우고 빈 데이터프레임으로 시작
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

        # 새 파일(혹은 방금 삭제한 파일) 기준으로 초기 데이터프레임 생성
        self.df = pd.DataFrame(
            {col: pd.Series(dtype=dtype) for col, dtype in self.column_types.items()}
        )
        self.save()

    def record_step(self, data: Dict[str, Any]):
        """기존 datetime 있으면 업데이트, 없으면 새로 추가"""
        dt = pd.to_datetime(data.get("datetime"))
        if dt is None:
            raise ValueError("datetime 값은 반드시 존재해야 합니다.")

        # datetime 기준으로 기존 행 있는지 확인
        existing_idx = self.df.index[self.df["datetime"] == dt]

        row = {}
        for col, dtype in self.column_types.items():
            val = data.get(col, None)
            try:
                if dtype.startswith("datetime") and val is not None:
                    row[col] = pd.to_datetime(val)
                else:
                    row[col] = pd.Series([val], dtype=dtype)[0]
            except Exception as e:
                raise ValueError(
                    f"[RecordManager] 컬럼 '{col}' 값 변환 실패: {val} → {dtype} / {e}"
                )

        import warnings

        warnings.filterwarnings("ignore")

        if not existing_idx.empty:
            # 이미 존재하면 해당 행 업데이트
            idx = existing_idx[0]
            for key, value in row.items():
                self.df.at[idx, key] = value
        else:
            # 존재하지 않으면 새 row 추가
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        self.save()

    def save(self):
        self.df = self.df.sort_values(by="datetime")  # 기본은 ascending=True → 오름차순
        self.df.to_csv(self.file_path, index=False, encoding="utf-8")

    def get_dataframe(self) -> pd.DataFrame:
        return self.df
