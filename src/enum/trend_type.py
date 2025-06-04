from enum import Enum


class TrendType(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"

    def __str__(self):
        return self.value
