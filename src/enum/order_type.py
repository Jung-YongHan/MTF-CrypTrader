from enum import Enum


class OrderType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

    def __str__(self):
        return self.value
