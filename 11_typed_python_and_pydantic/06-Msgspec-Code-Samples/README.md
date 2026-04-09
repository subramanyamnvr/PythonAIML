# Msgspec Code Samples

This folder introduces `msgspec`, a high-performance structured-data library designed for fast typed models and fast serialization.

## Why This Folder Exists

- `msgspec` overlaps with some Pydantic use cases, especially around typed payloads.
- It is valuable when performance matters a lot.
- Seeing `msgspec` beside Pydantic makes the tradeoff between speed and ecosystem ergonomics easier to understand.

## What You Should Learn Here

- how `msgspec.Struct` defines typed data models
- how JSON encoding and decoding works
- how `msgspec.convert(...)` turns plain data into typed objects
- how nested models and defaults behave

## Recommended Order

1. `01-basic_struct.py`
2. `02-json_encode_decode.py`
3. `03-conversion_and_validation.py`
4. `04-nested_structs_and_defaults.py`

## File Guide

### `01-basic_struct.py`

- Introduces a minimal `msgspec.Struct`.
- Best first example for understanding the shape of a msgspec model.

### `02-json_encode_decode.py`

- Shows serialization to bytes and decoding back into typed objects.
- Useful for API and messaging workloads.

### `03-conversion_and_validation.py`

- Demonstrates typed conversion from regular Python data.
- This is the closest file to a Pydantic-style parsing example.

### `04-nested_structs_and_defaults.py`

- Demonstrates nested objects, tuple defaults, and conversion to plain built-in types.
- Helps connect msgspec models to downstream Python code.

## Msgspec vs Pydantic

- Use `msgspec` when performance and fast serialization are top priorities.
- Use Pydantic when you want richer validation features, a larger ecosystem, and more familiar framework integration.
- `msgspec` is especially attractive for lean high-throughput services.

## Install Dependency

- `pip install -r requirements.txt`

## How To Run

- `python 01-basic_struct.py`
- `python 02-json_encode_decode.py`
- `python 03-conversion_and_validation.py`
- `python 04-nested_structs_and_defaults.py`
