import asyncio
from src.trading_system import TradingSystem


if __name__ == "__main__":
    asyncio.run(TradingSystem("bull", "BTC").run())
    # asyncio.run(TradingSystem("bear", "BTC").run())
    # asyncio.run(TradingSystem("total", "BTC").run())
