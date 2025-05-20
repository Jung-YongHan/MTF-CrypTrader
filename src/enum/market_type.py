from enum import Enum


class MarketType(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    TOTAL = "total"
    SIDEWAYS = "sideways"

    def __str__(self):
        return self.value
