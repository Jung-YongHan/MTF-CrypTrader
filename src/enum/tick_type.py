from enum import Enum


class TickType(str, Enum):
    MONTH1 = "month1"
    WEEK1 = "week1"
    DAY1 = "day1"
    HOUR4 = "hour4"
    HOUR1 = "hour1"
    MINUTE30 = "minute30"
    MINUTE15 = "minute15"
    MINUTE5 = "minute5"
    MINUTE3 = "minute3"
    MINUTE1 = "minute1"

    def __str__(self):
        return self.value

    @property
    def interval_in_minute(self) -> int:
        mapping = {
            TickType.MONTH1: 60 * 24 * 30,
            TickType.WEEK1: 60 * 24 * 7,
            TickType.DAY1: 60 * 24,
            TickType.HOUR4: 60 * 4,
            TickType.HOUR1: 60,
            TickType.MINUTE30: 30,
            TickType.MINUTE15: 15,
            TickType.MINUTE5: 5,
            TickType.MINUTE3: 3,
            TickType.MINUTE1: 1,
        }
        return mapping.get(self, 1440)

    @property
    def days(self) -> int:
        mapping = {
            TickType.MONTH1: 30,
            TickType.WEEK1: 7,
            TickType.DAY1: 1,
            TickType.HOUR4: 1 / 6,
            TickType.HOUR1: 1 / 24,
            TickType.MINUTE30: 1 / (30 * 24),
            TickType.MINUTE15: 1 / (15 * 24),
            TickType.MINUTE5: 1 / (5 * 24),
            TickType.MINUTE3: 1 / (3 * 24),
            TickType.MINUTE1: 1 / (1 * 24),
        }
        return mapping.get(self, 1)


def get_interval_in_minute(tick_type: TickType) -> int:
    """
    Get the interval in minutes for a given tick type.

    Args:
        tick_type (TickType): The tick type as a TickType enum.

    Returns:
        int: The interval in minutes.
    """
    return tick_type.interval_in_minute


if __name__ == "__main__":
    print(TickType.MINUTE1.days)
