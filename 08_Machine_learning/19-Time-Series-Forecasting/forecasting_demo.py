from __future__ import annotations

from dataclasses import dataclass
from math import sin, pi
from statistics import mean
from typing import Callable, Iterable, Sequence


Series = list[float]
ForecastFn = Callable[[Sequence[float], int], Series]


@dataclass
class BacktestResult:
    name: str
    forecasts: list[Series]
    actuals: list[Series]

    @property
    def mae(self) -> float:
        errors = [
            abs(pred - truth)
            for forecast, actual in zip(self.forecasts, self.actuals)
            for pred, truth in zip(forecast, actual)
        ]
        return mean(errors)


def seasonal_naive(history: Sequence[float], horizon: int, season_length: int = 7) -> Series:
    if len(history) < season_length:
        baseline = history[-1]
        return [baseline] * horizon
    return [history[-season_length + (step % season_length)] for step in range(horizon)]


def moving_average(history: Sequence[float], horizon: int, window: int = 5) -> Series:
    source = list(history[-window:]) if len(history) >= window else list(history)
    value = mean(source)
    return [value] * horizon


def drift_forecast(history: Sequence[float], horizon: int) -> Series:
    if len(history) < 2:
        return [history[-1]] * horizon
    slope = (history[-1] - history[0]) / (len(history) - 1)
    return [history[-1] + slope * step for step in range(1, horizon + 1)]


def rolling_origin_backtest(
    series: Sequence[float],
    horizon: int,
    min_train_size: int,
    forecast_fn: ForecastFn,
    name: str,
) -> BacktestResult:
    forecasts: list[Series] = []
    actuals: list[Series] = []
    for split in range(min_train_size, len(series) - horizon + 1):
        history = series[:split]
        actual = list(series[split : split + horizon])
        forecasts.append(forecast_fn(history, horizon))
        actuals.append(actual)
    return BacktestResult(name=name, forecasts=forecasts, actuals=actuals)


def generate_synthetic_series(length: int = 60) -> Series:
    values: Series = []
    for step in range(length):
        trend = 0.8 * step
        seasonality = 6.0 * sin((2 * pi * step) / 7)
        local_pattern = (step % 4) - 1.5
        values.append(round(40 + trend + seasonality + local_pattern, 2))
    return values


def future_forecasts(history: Sequence[float], horizon: int) -> dict[str, Series]:
    return {
        "seasonal_naive": seasonal_naive(history, horizon),
        "moving_average": moving_average(history, horizon),
        "drift": drift_forecast(history, horizon),
    }


def summarize(results: Iterable[BacktestResult]) -> None:
    ranked = sorted(results, key=lambda item: item.mae)
    print("Backtest leaderboard (lower MAE is better)")
    for result in ranked:
        print(f"- {result.name:16s} MAE={result.mae:.3f}")


def main() -> None:
    series = generate_synthetic_series()
    horizon = 3
    results = [
        rolling_origin_backtest(series, horizon, 20, seasonal_naive, "seasonal_naive"),
        rolling_origin_backtest(series, horizon, 20, moving_average, "moving_average"),
        rolling_origin_backtest(series, horizon, 20, drift_forecast, "drift"),
    ]
    summarize(results)
    print("\nFuture forecast from latest history window")
    for name, forecast in future_forecasts(series, horizon).items():
        print(f"- {name:16s} {forecast}")


if __name__ == "__main__":
    main()
