from enum import Enum


class MarketCategoryType(str, Enum):
    MACRO = "macro"
    MICRO = "micro"

    def __str__(self):
        return self.value
