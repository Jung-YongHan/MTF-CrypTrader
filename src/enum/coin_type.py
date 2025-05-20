from enum import Enum


class CoinType(str, Enum):
    BTC = "btc"

    def __str__(self):
        return self.value
