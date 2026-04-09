"""Lesson 4: frozen models, slots, and inheritance.

This file shows how dataclasses can be made immutable and memory-friendlier,
and how they behave in a small class hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Coordinates:
    x: float
    y: float


@dataclass(slots=True)
class Person:
    name: str


@dataclass(slots=True)
class Employee(Person):
    role: str


if __name__ == "__main__":
    point = Coordinates(10.0, 20.0)
    employee = Employee("Ava", "ML Engineer")
    print(point)
    print(employee)
