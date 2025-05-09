from src.trading_system import create_system

app = create_system(
    regime="bull",
    coin="btc",
    micro_tick=15,
)
if __name__ == "__main__":
    app.run()
