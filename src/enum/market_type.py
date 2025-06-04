from enum import Enum


class MarketType(str, Enum):
    BTC = "btc"

    def __str__(self):
        return self.value
