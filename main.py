from src.trading_system import create_system

# bull: '2023-10-01 09:00:00', '2024-04-01 09:00:00'
# bear: '2024-04-01 09:00:00', '2024-10-01 09:00:00'
# total: '2023-10-01 09:00:00', '2024-10-01 09:00:00'

app = create_system(
    regime="total",
    start_date="2023-10-01 09:00:00",
    end_date="2024-10-01 09:00:00",
    coin="btc",
    macro_tick="month1",
    micro_tick="day1",
    only_macro=True,
)

# app = create_system(
#     regime="total",
#     start_date="2023-10-01 09:00:00",
#     end_date="2024-09-30 09:00:00",
#     coin="btc",
#     macro_tick="month1",
#     micro_tick="day1",
# )

# app = create_system(
#     regime="total",
#     start_date="2023-10-01 09:00:00",
#     end_date="2024-09-30 09:00:00",
#     coin="btc",
#     macro_tick="month1",
#     micro_tick="day1",
# )
if __name__ == "__main__":
    app.run()
