# OnePydantic

`onepydantic` is a single practical Pydantic v2 project that pulls together many of the concepts spread across the rest of the module.

The example models an AI bootcamp enrollment flow, which is a good fit for Pydantic because the data comes from the outside world and should not be trusted until it is validated.

## Why This Project Exists

- it gives you one integrated example instead of many isolated snippets
- it shows how `typing` and `Annotated` power Pydantic models
- it demonstrates what happens when raw API-style input becomes safe Python objects
- it gives you a reusable project you can talk about in interviews, notes, or LinkedIn posts

## What This Project Covers

- `BaseModel` for structured validated data
- `ConfigDict` for strict model behavior like `extra="forbid"` and `validate_assignment=True`
- `Field(...)` for aliases, defaults, metadata, and numeric or length constraints
- `Annotated[...]` plus `StringConstraints` for reusable typed string rules
- nested models such as `Address`, `EmergencyContact`, `StudentProfile`, and `LearningModule`
- enums and `Literal` values for safe controlled choices
- specialized types like `EmailStr`, `HttpUrl`, `UUID`, `Decimal`, and `AwareDatetime`
- `field_validator` and `model_validator` for custom business rules
- `computed_field` for derived output such as `final_price`
- `field_serializer` for clean JSON output
- discriminated unions for different plan shapes
- `TypeAdapter` for validating lists of models
- `model_validate_json(...)`, `model_dump(...)`, `model_dump_json(...)`, and `model_json_schema()`

## Files

### `main.py`

The main runnable demo. It:

- validates JSON input from `sample_enrollment.json`
- parses it into nested Pydantic models
- serializes the result back into JSON-ready output
- shows computed values
- demonstrates assignment validation
- shows how validation errors are reported
- validates a batch of enrollments with `TypeAdapter`
- prints a small JSON schema snapshot

### `sample_enrollment.json`

A realistic external payload that uses alias-based field names such as `fullName`, `preferredTrack`, and `startAt`.

This helps show how Pydantic can accept API-style input while still exposing Python-friendly attribute names in code.

### `requirements.txt`

Minimal project-specific dependencies.

### `linkedin_post_outline.md`

A writing guide you can use to create a detailed LinkedIn post about `typing`, `from __future__ import annotations`, `dataclasses`, and Pydantic.

## Concept Map

### 1. Model Structure

`StudentProfile` and `BootcampEnrollment` show how `BaseModel` turns raw nested input into typed Python objects.

### 2. Typing + Annotated

Reusable type aliases such as `PhoneNumber`, `ModuleCode`, and `TagName` show how `typing` and `Annotated` keep model definitions clean.

### 3. Validation Rules

- field validators normalize names, coupon codes, notes, and tags
- model validators enforce business rules, such as requiring a portfolio for experienced learners

### 4. Safer Inputs

- `extra="forbid"` rejects unexpected fields
- `validate_assignment=True` keeps model instances valid even after creation

### 5. Serialization

- aliases let incoming JSON use camelCase while Python code uses snake_case
- custom serializers clean up datetime output for JSON responses
- `model_dump(mode="json", by_alias=True)` shows how to produce API-friendly output

### 6. Advanced Pydantic Concepts

- discriminated unions model `personal` versus `team` plans
- computed fields derive values without storing duplicate data
- `TypeAdapter` validates a list of enrollments in one step
- `model_json_schema()` shows schema generation

## How To Run

1. Open a terminal in this folder
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the demo:
   `python main.py`

## Suggested Learning Order

1. Read the JSON payload
2. Scan the reusable annotated types
3. Read the models from smallest to largest
4. Run the script and inspect the output
5. Compare what the project validates automatically versus what a plain dataclass would not

## Best Takeaway

Pydantic is strongest when your data enters the system from an API, form, file, LLM response, or config source.

If the data is already trusted and fully internal, `dataclasses` may be enough.
