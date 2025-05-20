from enum import Enum


class PulseType(str, Enum):
    BULL = "상승 돌파"
    BEAR = "하락 돌파"
    SIDEWAYS = "돌파 없음"

    def __str__(self):
        return self.value
