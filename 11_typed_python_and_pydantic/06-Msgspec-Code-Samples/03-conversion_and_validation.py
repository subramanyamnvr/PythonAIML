"""Lesson 3: converting raw data into typed structs.

This file is the closest msgspec example to a Pydantic parsing workflow.
"""

from __future__ import annotations

import msgspec


class Product(msgspec.Struct):
    sku: str
    price: float
    tags: list[str]


if __name__ == "__main__":
    raw_payload = {
        "sku": "BK-100",
        "price": "19.99",
        "tags": ["books", "python"],
    }
    # convert applies msgspec's typed conversion rules to regular Python data.
    product = msgspec.convert(raw_payload, type=Product)

    print(product)
