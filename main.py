import json

from src.trading_system import create_system


def run_backtest(config: dict) -> dict:
    app = create_system(**config)
    return app.run()


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        test_configs = json.load(f)

    results = []
    for cfg in test_configs:
        perf = run_backtest(cfg)
        results.append({**cfg, "performance": perf})

    print("==== All Results ====")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
