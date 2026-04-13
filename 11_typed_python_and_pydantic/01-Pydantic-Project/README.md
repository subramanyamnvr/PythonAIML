# Pydantic Project

This folder now includes a concrete project you can run while learning the wider Pydantic path.

## Included Project

### `onepydantic`

An all-in-one Pydantic v2 learning project built around an AI bootcamp enrollment workflow.

It demonstrates:

- `BaseModel` and nested models
- `Field(...)` metadata, defaults, and constraints
- `ConfigDict` for stricter model behavior
- field aliases for validation and serialization
- specialized types such as `EmailStr`, `HttpUrl`, `UUID`, `Decimal`, and timezone-aware datetimes
- `Annotated` plus string constraints
- field validators and model validators
- computed fields and custom serialization
- discriminated unions for plan types
- `TypeAdapter` for validating collections
- JSON schema generation

## Folder Guide

- `onepydantic/README.md`: guided explanation of the project and concept map
- `onepydantic/main.py`: runnable integrated demo
- `onepydantic/sample_enrollment.json`: external payload used by the demo
- `onepydantic/requirements.txt`: minimal dependency list
- `onepydantic/linkedin_post_outline.md`: detailed LinkedIn post format and outline for `typing`, `__future__`, `dataclasses`, and Pydantic

## How To Use This Folder

1. Read `onepydantic/README.md`
2. Run `python main.py` inside `onepydantic`
3. Compare the integrated project with the later numbered Pydantic notebooks
4. Use `linkedin_post_outline.md` when you are ready to turn the learning into a post
