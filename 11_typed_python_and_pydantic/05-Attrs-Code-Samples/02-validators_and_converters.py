"""Lesson 2: validators and converters.

This file demonstrates the biggest quality-of-life improvement attrs offers:
- converters clean incoming values
- validators enforce rules at construction time
"""

from __future__ import annotations

from attrs import define, field


def non_empty_string(instance: object, attribute: object, value: str) -> None:
    if not value:
        raise ValueError("value must not be empty")


def positive_number(instance: object, attribute: object, value: float) -> None:
    if value <= 0:
        raise ValueError("value must be positive")


@define
class Product:
    # converter normalizes raw input before validation runs.
    sku: str = field(converter=lambda value: str(value).strip(), validator=non_empty_string)
    price: float = field(converter=float, validator=positive_number)


if __name__ == "__main__":
    product = Product(" BK-100 ", "19.99")
    print(product)
