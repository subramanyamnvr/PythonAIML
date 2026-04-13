from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class EvalCase:
    prompt: str
    response: str
    expected: str


@dataclass
class EvalScore:
    name: str
    value: float


@dataclass
class Span:
    name: str
    duration_ms: float
    metadata: dict[str, str]


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def trace(self, name: str, **metadata: str):
        start = perf_counter()
        yield
        duration_ms = (perf_counter() - start) * 1000
        self.spans.append(Span(name=name, duration_ms=duration_ms, metadata=metadata))


def exact_match(case: EvalCase) -> EvalScore:
    return EvalScore(name="exact_match", value=1.0 if case.response.strip().lower() == case.expected.strip().lower() else 0.0)


def contains_expected(case: EvalCase) -> EvalScore:
    return EvalScore(name="contains_expected", value=1.0 if case.expected.lower() in case.response.lower() else 0.0)


def summarize(scores: list[EvalScore]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for score in scores:
        buckets.setdefault(score.name, []).append(score.value)
    return {name: sum(values) / len(values) for name, values in buckets.items()}


def main() -> None:
    cases = [
        EvalCase("Capital of France?", "Paris", "Paris"),
        EvalCase("Return JSON with key status", '{"status": "ok"}', "status"),
        EvalCase("What tool should be used for retrieval?", "Use a vector search retriever.", "retriever"),
    ]

    tracer = Tracer()
    scores: list[EvalScore] = []
    with tracer.trace("eval_suite", suite="starter"):
        for case in cases:
            with tracer.trace("eval_case", prompt=case.prompt):
                scores.append(exact_match(case))
                scores.append(contains_expected(case))

    print("Aggregate scores")
    for name, value in summarize(scores).items():
        print(f"- {name:18s} {value:.2f}")

    print("\nRecorded spans")
    for span in tracer.spans:
        print(f"- {span.name:10s} {span.duration_ms:7.3f}ms metadata={span.metadata}")


if __name__ == "__main__":
    main()
