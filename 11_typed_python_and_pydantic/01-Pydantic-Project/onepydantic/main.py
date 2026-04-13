"""OnePydantic: an integrated Pydantic v2 learning project.

This file demonstrates many day-to-day Pydantic concepts in one place:
- nested models and reusable annotated types
- aliases, constraints, specialized types, and defaults
- field and model validators
- computed fields and custom JSON serialization
- discriminated unions and batch validation with TypeAdapter
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    HttpUrl,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PhoneNumber = Annotated[str, StringConstraints(pattern=r"^\+\d{10,15}$")]
ModuleCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2,4}-\d{3}$")]
TagName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=20, pattern=r"^[a-z0-9-]+$"),
]


class Track(str, Enum):
    GENAI = "genai"
    MLOPS = "mlops"
    CV = "computer-vision"


class DeliveryMode(str, Enum):
    LIVE = "live"
    HYBRID = "hybrid"


class Address(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    line1: NonEmptyStr
    city: NonEmptyStr
    state: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=30)]
    postal_code: Annotated[str, StringConstraints(pattern=r"^\d{6}$")] = Field(alias="postalCode")
    country: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=30)] = "India"


class EmergencyContact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: NonEmptyStr
    relationship: Annotated[str, StringConstraints(min_length=2, max_length=20)]
    phone: PhoneNumber


class LearningModule(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: ModuleCode
    title: Annotated[str, StringConstraints(min_length=3, max_length=80)]
    duration_hours: int = Field(gt=0, le=40)
    tags: list[TagName] = Field(default_factory=list, max_length=5)

    @field_validator("tags")
    @classmethod
    def unique_tags(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class PersonalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_type: Literal["personal"]
    seat_count: Literal[1] = 1
    mentoring_sessions: int = Field(ge=0, le=5, default=1)


class TeamPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    plan_type: Literal["team"]
    seat_count: int = Field(ge=2, le=25)
    company_name: Annotated[str, StringConstraints(min_length=2, max_length=80)]
    mentoring_sessions: int = Field(ge=1, le=20, default=4)


EnrollmentPlan = Annotated[PersonalPlan | TeamPlan, Field(discriminator="plan_type")]


class StudentProfile(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    student_id: UUID = Field(default_factory=uuid4, alias="studentId")
    full_name: Annotated[str, StringConstraints(min_length=3, max_length=60)] = Field(alias="fullName")
    email: EmailStr
    experience_years: int = Field(ge=0, le=25, alias="experienceYears")
    preferred_track: Track = Field(alias="preferredTrack")
    portfolio_url: HttpUrl | None = Field(default=None, alias="portfolioUrl")
    joined_on: date = Field(default_factory=date.today, alias="joinedOn")
    address: Address
    emergency_contact: EmergencyContact = Field(alias="emergencyContact")

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str) -> str:
        return " ".join(part.capitalize() for part in value.split())

    @model_validator(mode="after")
    def require_portfolio_for_experienced_students(self) -> StudentProfile:
        if self.experience_years >= 2 and self.portfolio_url is None:
            raise ValueError("portfolioUrl is required when experienceYears is 2 or more")
        return self


class BootcampEnrollment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, str_strip_whitespace=True)

    enrollment_id: UUID = Field(default_factory=uuid4, alias="enrollmentId")
    student: StudentProfile
    modules: list[LearningModule] = Field(min_length=1, max_length=6)
    plan: EnrollmentPlan
    delivery_mode: DeliveryMode = Field(alias="deliveryMode")
    start_at: AwareDatetime = Field(alias="startAt")
    coupon_code: Annotated[str | None, StringConstraints(strip_whitespace=True, min_length=5, max_length=12)] = (
        Field(default=None, alias="couponCode")
    )
    base_price: Decimal = Field(default=Decimal("14999.00"), gt=0, alias="basePrice")
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="createdAt")
    notes: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("notes", mode="before")
    @classmethod
    def split_pipe_separated_notes(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split("|") if part.strip()]
        return value

    @field_validator("coupon_code")
    @classmethod
    def normalize_coupon_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.upper()

    @field_validator("modules")
    @classmethod
    def enforce_unique_module_codes(cls, value: list[LearningModule]) -> list[LearningModule]:
        codes = [module.code for module in value]
        if len(codes) != len(set(codes)):
            raise ValueError("module codes must be unique inside one enrollment")
        return value

    @model_validator(mode="after")
    def validate_live_team_rule(self) -> BootcampEnrollment:
        if self.plan.plan_type == "team" and self.delivery_mode == DeliveryMode.LIVE and self.plan.seat_count < 3:
            raise ValueError("live team enrollments require at least 3 seats")
        return self

    @computed_field(alias="totalLearningHours", return_type=int)
    @property
    def total_learning_hours(self) -> int:
        return sum(module.duration_hours for module in self.modules)

    @computed_field(alias="finalPrice", return_type=Decimal)
    @property
    def final_price(self) -> Decimal:
        subtotal = self.base_price * Decimal(self.plan.seat_count)
        if self.coupon_code == "LAUNCH10":
            subtotal *= Decimal("0.90")
        return subtotal.quantize(Decimal("0.01"))

    @computed_field(alias="cohortLabel", return_type=str)
    @property
    def cohort_label(self) -> str:
        return f"{self.delivery_mode.value}-{self.start_at.date().isoformat()}"

    @field_serializer("start_at", "created_at", when_used="json")
    def serialize_datetimes(self, value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_sample_json() -> str:
    return Path(__file__).with_name("sample_enrollment.json").read_text(encoding="utf-8")


def load_sample_payload() -> dict[str, object]:
    return json.loads(load_sample_json())


def build_second_payload() -> dict[str, object]:
    payload = load_sample_payload()
    payload["student"] = {
        **payload["student"],
        "fullName": "neha kapoor",
        "email": "neha@example.com",
        "experienceYears": 1,
        "portfolioUrl": None,
    }
    payload["plan"] = {"plan_type": "personal", "seat_count": 1, "mentoring_sessions": 2}
    payload["deliveryMode"] = "live"
    payload["couponCode"] = "focus5"
    return payload


def print_section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def demo_success_case() -> BootcampEnrollment:
    print_section("VALIDATING JSON INPUT")
    enrollment = BootcampEnrollment.model_validate_json(load_sample_json())
    print(enrollment)

    print_section("MODEL_DUMP JSON-READY OUTPUT")
    print(json.dumps(enrollment.model_dump(by_alias=True, mode="json"), indent=2))

    print_section("MODEL_DUMP_JSON OUTPUT")
    print(enrollment.model_dump_json(by_alias=True, indent=2))

    print_section("COMPUTED FIELDS")
    print(
        {
            "total_learning_hours": enrollment.total_learning_hours,
            "final_price": str(enrollment.final_price),
            "cohort_label": enrollment.cohort_label,
        }
    )

    return enrollment


def demo_assignment_validation(enrollment: BootcampEnrollment) -> None:
    print_section("ASSIGNMENT VALIDATION")
    try:
        enrollment.student.experience_years = -1
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        print(first_error)


def demo_validation_errors() -> None:
    print_section("VALIDATION ERRORS")
    bad_payload = load_sample_payload()
    bad_payload["student"]["email"] = "not-an-email"
    bad_payload["student"]["unknownField"] = "should fail"
    bad_payload["modules"][0]["duration_hours"] = 0
    bad_payload["deliveryMode"] = "live"
    bad_payload["plan"] = {
        "plan_type": "team",
        "seat_count": 2,
        "company_name": "Acme AI Labs",
        "mentoring_sessions": 4,
    }

    try:
        BootcampEnrollment.model_validate(bad_payload)
    except ValidationError as exc:
        for error in exc.errors(include_url=False):
            print(f"loc={error['loc']} msg={error['msg']}")


def demo_business_rule_error() -> None:
    print_section("MODEL VALIDATOR BUSINESS RULE")
    rule_breaking_payload = load_sample_payload()
    rule_breaking_payload["deliveryMode"] = "live"
    rule_breaking_payload["plan"] = {
        "plan_type": "team",
        "seat_count": 2,
        "company_name": "Acme AI Labs",
        "mentoring_sessions": 4,
    }

    try:
        BootcampEnrollment.model_validate(rule_breaking_payload)
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        print(first_error)


def demo_batch_validation() -> None:
    print_section("TYPEADAPTER BATCH VALIDATION")
    batch_adapter = TypeAdapter(list[BootcampEnrollment])
    enrollments = batch_adapter.validate_python([load_sample_payload(), build_second_payload()])
    print(f"Validated {len(enrollments)} enrollments in one call.")
    print([enrollment.student.full_name for enrollment in enrollments])


def demo_schema_snapshot() -> None:
    print_section("JSON SCHEMA SNAPSHOT")
    schema = BootcampEnrollment.model_json_schema(by_alias=True)
    print(
        {
            "title": schema.get("title"),
            "top_level_properties": sorted(schema.get("properties", {}).keys()),
        }
    )


if __name__ == "__main__":
    validated_enrollment = demo_success_case()
    demo_assignment_validation(validated_enrollment)
    demo_validation_errors()
    demo_business_rule_error()
    demo_batch_validation()
    demo_schema_snapshot()
