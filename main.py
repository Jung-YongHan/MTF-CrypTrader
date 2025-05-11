from src.trading_system import create_system

# bull: '2024-11-05 09:00:00', '2025-01-20 09:00:00'
# bear: '2025-01-21 09:00:00', '2025-04-08 09:00:00'
# bull: '2024-11-05 09:00:00', '2025-01-20 09:00:00'

app = create_system(
    regime="bull",
    start_date="2024-11-05 09:00:00",
    end_date="2025-01-20 09:00:00",
    coin="btc",
    micro_tick=15,
)
if __name__ == "__main__":
    app.run()
