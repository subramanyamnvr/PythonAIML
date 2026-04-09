# Attrs Code Samples

This folder introduces `attrs`, a popular library for building structured Python classes with a nicer developer experience than plain classes and more flexibility than basic dataclasses.

## Why This Folder Exists

- `attrs` sits in the middle ground between `dataclasses` and Pydantic.
- It is useful when you want clean model classes plus validators, converters, immutability, and factories.
- Learning it helps you understand the broader Python model ecosystem, not just Pydantic.

## What You Should Learn Here

- how `@define` creates compact model classes
- how `field(...)` enables validators, converters, and defaults
- how frozen models behave
- how nested `attrs` models serialize to dictionaries

## Recommended Order

1. `01-basic_attrs_model.py`
2. `02-validators_and_converters.py`
3. `03-frozen_slots_and_factories.py`
4. `04-nested_models_and_serialization.py`

## File Guide

### `01-basic_attrs_model.py`

- The smallest useful `attrs` model.
- Good for comparing side by side with a dataclass.

### `02-validators_and_converters.py`

- The key `attrs` feature demo.
- Shows automatic cleaning with converters and rule checks with validators.

### `03-frozen_slots_and_factories.py`

- Demonstrates immutable configuration objects.
- Uses a factory for safe defaults.

### `04-nested_models_and_serialization.py`

- Shows nested models and conversion to plain Python dictionaries.
- Useful for understanding app-level object modeling before full schema frameworks.

## Attrs vs Pydantic

- Use `attrs` when you want elegant Python objects with validation hooks but do not need full schema parsing.
- Use Pydantic when input comes from outside your code and must be parsed, coerced, and validated rigorously.
- `attrs` is object-model first; Pydantic is input-validation first.

## Install Dependency

- `pip install -r requirements.txt`

## How To Run

- `python 01-basic_attrs_model.py`
- `python 02-validators_and_converters.py`
- `python 03-frozen_slots_and_factories.py`
- `python 04-nested_models_and_serialization.py`
