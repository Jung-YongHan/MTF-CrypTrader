from src.trading_system import create_system

# month1, day1
# bull: '2023-10-01 09:00:00', '2024-04-01 09:00:00'
# bear: '2024-04-01 09:00:00', '2024-10-01 09:00:00'
# total: '2023-10-01 09:00:00', '2024-10-01 09:00:00'

# day1, hour1
# bull: '2024-11-10 09:00:00', '2024-12-17 09:00:00'
# bear: '2025-01-21 09:00:00', '2025-02-27 09:00:00'
# total: '2024-11-10 09:00:00', '2025-02-27 09:00:00'

app = create_system(
    regime="total",
    start_date="2023-10-01 09:00:00",
    end_date="2024-10-01 09:00:00",
    coin="btc",
    macro_tick="month1",
    micro_tick="day1",
    only_macro=False,
)
if __name__ == "__main__":
    app.run()
