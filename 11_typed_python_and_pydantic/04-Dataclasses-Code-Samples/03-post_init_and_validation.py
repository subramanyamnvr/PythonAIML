"""Lesson 3: post-init hooks and derived fields.

This file shows how dataclasses can perform simple validation
after object creation and compute a field from other values.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LineItem:
    name: str
    unit_price: float
    quantity: int = 1
    # init=False means callers do not pass this directly.
    subtotal: float = field(init=False)

    def __post_init__(self) -> None:
        # __post_init__ runs after the generated __init__ method finishes.
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.unit_price <= 0:
            raise ValueError("unit_price must be positive")
        self.subtotal = self.unit_price * self.quantity


if __name__ == "__main__":
    item = LineItem("Mechanical Keyboard", 99.0, 2)
    print(item.subtotal)
