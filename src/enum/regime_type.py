from enum import Enum


class RegimeType(str, Enum):
    BULL = "상승장"
    BEAR = "하락장"
    SIDEWAYS = "횡보장"

    def __str__(self):
        return self.value
