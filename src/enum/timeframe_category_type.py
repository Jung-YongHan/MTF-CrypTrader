from enum import Enum


class TimeframeCategoryType(str, Enum):
    HIGHER = "higher"
    LOWER = "lower"

    def __str__(self):
        return self.value
