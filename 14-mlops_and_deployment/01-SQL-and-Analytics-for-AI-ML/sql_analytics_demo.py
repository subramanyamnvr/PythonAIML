from __future__ import annotations

import sqlite3


def build_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute(
        """
        create table events (
            user_id text,
            signup_week text,
            viewed_pricing integer,
            started_trial integer,
            converted integer,
            feature_score real
        )
        """
    )
    cursor.executemany(
        "insert into events values (?, ?, ?, ?, ?, ?)",
        [
            ("u1", "2026-W13", 1, 1, 1, 0.91),
            ("u2", "2026-W13", 1, 1, 0, 0.74),
            ("u3", "2026-W13", 1, 0, 0, 0.65),
            ("u4", "2026-W14", 1, 1, 1, 0.88),
            ("u5", "2026-W14", 0, 0, 0, 0.42),
            ("u6", "2026-W14", 1, 1, 0, 0.79),
        ],
    )
    connection.commit()
    return connection


def print_query(title: str, connection: sqlite3.Connection, query: str) -> None:
    cursor = connection.execute(query)
    print(title)
    columns = [description[0] for description in cursor.description]
    print("-", columns)
    for row in cursor.fetchall():
        print("-", row)
    print()


def main() -> None:
    connection = build_database()
    print_query(
        "Weekly funnel",
        connection,
        """
        select
            signup_week,
            count(*) as signups,
            sum(viewed_pricing) as pricing_views,
            sum(started_trial) as trials,
            sum(converted) as conversions
        from events
        group by signup_week
        order by signup_week
        """,
    )
    print_query(
        "Conversion rate by cohort",
        connection,
        """
        select
            signup_week,
            round(avg(converted), 3) as conversion_rate,
            round(avg(feature_score), 3) as average_feature_score
        from events
        group by signup_week
        order by signup_week
        """,
    )
    print_query(
        "Data quality checks",
        connection,
        """
        select
            count(*) as total_rows,
            sum(case when feature_score is null then 1 else 0 end) as null_feature_scores,
            sum(case when converted = 1 and started_trial = 0 then 1 else 0 end) as invalid_conversion_rows
        from events
        """,
    )


if __name__ == "__main__":
    main()
