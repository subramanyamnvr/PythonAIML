from __future__ import annotations

from collections import deque
from statistics import mean


class LatencyMonitor:
    def __init__(self, window_size: int = 5) -> None:
        self.samples: deque[float] = deque(maxlen=window_size)

    def record(self, latency_ms: float) -> None:
        self.samples.append(latency_ms)

    def summary(self) -> dict[str, float]:
        ordered = sorted(self.samples)
        return {
            "avg_ms": round(mean(self.samples), 2),
            "p95_ms": round(ordered[max(0, int(0.95 * (len(ordered) - 1)))], 2),
            "max_ms": round(max(ordered), 2),
        }


def psi(expected: list[float], actual: list[float], bins: list[float]) -> float:
    def bucketize(values: list[float]) -> list[int]:
        counts = [0] * (len(bins) - 1)
        for value in values:
            for index in range(len(bins) - 1):
                left, right = bins[index], bins[index + 1]
                if left <= value < right or (index == len(bins) - 2 and value == right):
                    counts[index] += 1
                    break
        return counts

    expected_counts = bucketize(expected)
    actual_counts = bucketize(actual)
    total_expected = sum(expected_counts)
    total_actual = sum(actual_counts)
    score = 0.0
    for exp_count, act_count in zip(expected_counts, actual_counts):
        exp_ratio = max(exp_count / total_expected, 1e-6)
        act_ratio = max(act_count / total_actual, 1e-6)
        score += (act_ratio - exp_ratio) * __import__("math").log(act_ratio / exp_ratio)
    return score


def main() -> None:
    monitor = LatencyMonitor()
    for latency in [82.5, 77.0, 91.2, 103.8, 88.6]:
        monitor.record(latency)
    print("Latency summary")
    print(monitor.summary())

    baseline_scores = [0.12, 0.18, 0.24, 0.31, 0.44, 0.51, 0.57, 0.63, 0.71]
    live_scores = [0.08, 0.11, 0.15, 0.19, 0.22, 0.28, 0.37, 0.45, 0.49]
    score = psi(baseline_scores, live_scores, bins=[0.0, 0.2, 0.4, 0.6, 0.8])
    print("\nDrift check")
    print(f"- psi={score:.4f}")
    print(f"- status={'alert' if score > 0.2 else 'stable'}")


if __name__ == "__main__":
    main()
