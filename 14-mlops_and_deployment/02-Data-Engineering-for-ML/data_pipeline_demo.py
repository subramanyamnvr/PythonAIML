from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class RawRecord:
    user_id: str
    country: str
    sessions: int
    purchases: int
    revenue: float


def validate_record(record: RawRecord) -> list[str]:
    issues: list[str] = []
    if not record.user_id:
        issues.append("missing_user_id")
    if record.sessions < 0:
        issues.append("negative_sessions")
    if record.purchases < 0:
        issues.append("negative_purchases")
    if record.revenue < 0:
        issues.append("negative_revenue")
    if record.purchases > record.sessions:
        issues.append("purchases_exceed_sessions")
    return issues


def build_features(records: list[RawRecord]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for record in records:
        conversion_rate = record.purchases / record.sessions if record.sessions else 0.0
        average_order_value = record.revenue / record.purchases if record.purchases else 0.0
        rows.append(
            {
                "user_id": record.user_id,
                "country": record.country.upper(),
                "sessions": record.sessions,
                "purchases": record.purchases,
                "conversion_rate": round(conversion_rate, 3),
                "average_order_value": round(average_order_value, 2),
                "high_value_user": 1 if record.revenue >= 250 else 0,
            }
        )
    return rows


def main() -> None:
    raw_records = [
        RawRecord("u1", "in", 12, 2, 140.0),
        RawRecord("u2", "us", 4, 1, 310.0),
        RawRecord("u3", "uk", 0, 0, 0.0),
        RawRecord("u4", "in", 9, 3, 420.0),
        RawRecord("u5", "de", 5, 7, 210.0),
    ]

    valid_records: list[RawRecord] = []
    rejected: list[tuple[str, list[str]]] = []
    for record in raw_records:
        issues = validate_record(record)
        if issues:
            rejected.append((record.user_id, issues))
        else:
            valid_records.append(record)

    features = build_features(valid_records)
    print("Accepted feature rows")
    for row in features:
        print("-", row)

    print("\nRejected records")
    for user_id, issues in rejected:
        print(f"- {user_id}: {issues}")

    if features:
        print("\nPipeline summary")
        print(f"- average_conversion_rate={mean(row['conversion_rate'] for row in features):.3f}")
        print(f"- high_value_users={sum(int(row['high_value_user']) for row in features)}")


if __name__ == "__main__":
    main()
