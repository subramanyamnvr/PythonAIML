"""Lesson 1: basic type hints.

This file shows the core idea behind typing:
- annotate function inputs and outputs
- use broad container interfaces like Iterable when possible
- return concrete result types when the function builds a new value
"""

from __future__ import annotations

from typing import Iterable


def normalize_tags(tags: Iterable[str]) -> list[str]:
    # Accept any iterable of strings, not just a list.
    return sorted({tag.strip().lower() for tag in tags if tag.strip()})


def average_score(scores: list[float]) -> float:
    # Return type hints make the function contract explicit.
    return sum(scores) / len(scores)


if __name__ == "__main__":
    cleaned = normalize_tags(["  Python", "pydantic ", "python", "", "Typing"])
    average = average_score([8.5, 9.0, 10.0])

    print(cleaned)
    print(average)
