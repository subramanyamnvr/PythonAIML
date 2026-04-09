"""Lesson 2: defaults and ordering.

This file highlights two common dataclass features:
- ordered comparisons for sorting
- safe mutable defaults with default_factory
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(order=True)
class Task:
    priority: int
    title: str
    # default_factory creates a fresh list per instance.
    labels: list[str] = field(default_factory=list)


if __name__ == "__main__":
    tasks = [
        Task(2, "Write notes", ["study"]),
        Task(1, "Run examples", ["practice"]),
    ]
    print(sorted(tasks))
