from __future__ import annotations


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def bucket_age(age: int) -> str:
    if age < 25:
        return "young"
    if age < 40:
        return "mid"
    return "senior"


def build_feature_row(name: str, age: int, monthly_spend: float, country: str) -> dict[str, int | float | str]:
    normalized_country = normalize_text(country).upper()
    spend_per_year = round(monthly_spend * 12, 2)
    return {
        "customer_name": normalize_text(name),
        "age": age,
        "age_bucket": bucket_age(age),
        "monthly_spend": monthly_spend,
        "annualized_spend": spend_per_year,
        "country": normalized_country,
        "is_high_value": 1 if monthly_spend >= 250 else 0,
    }
