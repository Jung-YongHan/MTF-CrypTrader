from enum import Enum


class RecordType(str, Enum):
    MACRO = "macro"
    MICRO = "micro"
    TRADE = "trade"

    def __str__(self):
        return self.value
