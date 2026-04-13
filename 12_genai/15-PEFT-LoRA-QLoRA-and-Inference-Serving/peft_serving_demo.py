from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest


Matrix = list[list[float]]


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    transposed = list(zip(*right))
    return [[sum(a * b for a, b in zip(row, column)) for column in transposed] for row in left]


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return [[a + b for a, b in zip(row_left, row_right)] for row_left, row_right in zip(left, right)]


def quantize(values: Matrix, scale: float = 0.1) -> list[list[int]]:
    return [[round(value / scale) for value in row] for row in values]


@dataclass
class LoRAAdapter:
    a: Matrix
    b: Matrix
    alpha: float
    rank: int

    def delta(self) -> Matrix:
        base = matrix_multiply(self.a, self.b)
        factor = self.alpha / self.rank
        return [[factor * value for value in row] for row in base]


class InferenceBatcher:
    def __init__(self, max_batch_size: int = 3) -> None:
        self.max_batch_size = max_batch_size
        self.queue: list[list[float]] = []

    def enqueue(self, request: list[float]) -> list[list[float]] | None:
        self.queue.append(request)
        if len(self.queue) >= self.max_batch_size:
            return self.flush()
        return None

    def flush(self) -> list[list[float]]:
        batch = self.queue[:]
        self.queue.clear()
        return batch


def linear_forward(inputs: Matrix, weights: Matrix) -> Matrix:
    return matrix_multiply(inputs, weights)


def main() -> None:
    base_weights = [[0.4, 0.1], [0.2, 0.3], [0.5, 0.7]]
    adapter = LoRAAdapter(
        a=[[0.2, -0.1], [0.0, 0.3], [0.1, 0.2]],
        b=[[0.4, 0.5], [-0.2, 0.1]],
        alpha=8.0,
        rank=2,
    )
    adapted_weights = matrix_add(base_weights, adapter.delta())
    print("Merged LoRA weights")
    for row in adapted_weights:
        print("-", [round(value, 4) for value in row], "quantized=", quantize([row])[0])

    batcher = InferenceBatcher(max_batch_size=2)
    requests = [[1.0, 0.5, 0.2], [0.8, 0.1, 0.9], [0.2, 0.4, 0.7]]
    for request in requests:
        batch = batcher.enqueue(request)
        if batch is not None:
            print("\nServed batch")
            for prediction in linear_forward(batch, adapted_weights):
                print("-", [round(value, 4) for value in prediction])

    remaining = batcher.flush()
    if remaining:
        print("\nFinal partial batch")
        for prediction in linear_forward(remaining, adapted_weights):
            print("-", [round(value, 4) for value in prediction])


if __name__ == "__main__":
    main()
