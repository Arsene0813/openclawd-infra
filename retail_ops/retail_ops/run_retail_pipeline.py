from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SQL_DIR = BASE_DIR / "sql"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_CSV = DATA_DIR / "store_monthly_metrics_sample.csv"


TABLE_SCHEMA = """
CREATE TABLE store_monthly_metrics (
    store_id TEXT,
    period TEXT,
    region_type TEXT,
    store_type TEXT,
    business_district_rank INTEGER,
    revenue REAL,
    valid_orders INTEGER,
    avg_order_value REAL,
    store_visitors INTEGER,
    order_conversion_rate REAL,
    search_exposure INTEGER,
    search_visitors INTEGER,
    refund_amount REAL,
    refund_orders INTEGER,
    promotion_gmv REAL,
    merchant_subsidy REAL,
    top10_products TEXT
);
"""


NUMERIC_FIELDS = {
    "business_district_rank": int,
    "revenue": float,
    "valid_orders": int,
    "avg_order_value": float,
    "store_visitors": int,
    "order_conversion_rate": float,
    "search_exposure": int,
    "search_visitors": int,
    "refund_amount": float,
    "refund_orders": int,
    "promotion_gmv": float,
    "merchant_subsidy": float,
}


def load_store_metrics(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS store_monthly_metrics;")
    conn.execute(TABLE_SCHEMA)

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []

        for row in reader:
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                if key in NUMERIC_FIELDS:
                    cleaned[key] = NUMERIC_FIELDS[key](value)
                else:
                    cleaned[key] = value
            rows.append(cleaned)

    if not rows:
        raise ValueError("No rows found in input CSV.")

    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)

    conn.executemany(
        f"INSERT INTO store_monthly_metrics ({column_sql}) VALUES ({placeholders})",
        [[row[col] for col in columns] for row in rows],
    )
    conn.commit()


def run_sql_file(conn: sqlite3.Connection, sql_filename: str) -> tuple[list[str], list[dict[str, Any]]]:
    sql = (SQL_DIR / sql_filename).read_text(encoding="utf-8")
    cursor = conn.execute(sql)
    columns = [description[0] for description in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return columns, rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def money(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.2f}"


def build_memory_facts(derived_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    search_heavy_count = sum(
        1 for row in derived_rows if float(row["search_visit_share"]) >= 0.85
    )
    total_stores = len(derived_rows)

    reference_store = max(
        derived_rows,
        key=lambda row: (
            float(row["revenue"]),
            int(row["valid_orders"]),
            float(row["search_exposure_to_visit_rate"]),
            float(row["revenue_per_visitor"]),
        ),
    )

    facts: list[dict[str, Any]] = [
        {
            "type": "cross_store_pattern",
            "period": "2026-03",
            "scope": f"{total_stores} anonymized Meituan stores",
            "value": (
                f"{search_heavy_count} out of {total_stores} stores show search visit share "
                "above 85%, suggesting that search-driven traffic is a major entry source "
                "in the current public sample."
            ),
            "decision_use": (
                "Search visibility should remain an important operational focus, but it "
                "should not be used as the only explanation for store performance."
            ),
            "confidence": "medium",
            "source": "retail_ops/sql/01_derived_metrics.sql",
        },
        {
            "type": "store_metric_signal",
            "period": "2026-03",
            "store_id": reference_store["store_id"],
            "value": (
                f"Store {reference_store['store_id']} has the highest revenue, high valid order count, "
                f"search exposure-to-visit rate of {pct(reference_store['search_exposure_to_visit_rate'])}, "
                f"and revenue per visitor of {money(reference_store['revenue_per_visitor'])} "
                "in the current public sample."
            ),
            "decision_use": (
                "This store can be reviewed as a reference store, but its strategy should "
                "not be copied directly without checking region, product mix, promotion "
                "structure, fulfillment conditions, and customer profile."
            ),
            "confidence": "medium",
            "source": "retail_ops/sql/02_cross_store_ranking.sql",
        },
    ]

    for row in derived_rows:
        refund_pressure = (
            float(row["refund_revenue_ratio"]) >= 0.10
            or float(row["refund_order_ratio"]) >= 0.12
        )
        if refund_pressure:
            facts.append(
                {
                    "type": "risk_signal",
                    "period": row["period"],
                    "store_id": row["store_id"],
                    "value": (
                        f"Store {row['store_id']} shows refund pressure in the current sample: "
                        f"refund revenue ratio is {pct(row['refund_revenue_ratio'])}, "
                        f"and refund order ratio is {pct(row['refund_order_ratio'])}."
                    ),
                    "decision_use": (
                        "Before increasing traffic investment, review refund reasons, "
                        "product description clarity, pricing, promotion structure, and "
                        "fulfillment reliability."
                    ),
                    "confidence": "medium",
                    "source": "retail_ops/sql/03_conservative_store_tags.sql",
                }
            )

    for row in derived_rows:
        high_subsidy = float(row["subsidy_to_revenue_ratio"]) >= 0.40
        low_refund = float(row["refund_revenue_ratio"]) < 0.10

        if high_subsidy and low_refund:
            facts.append(
                {
                    "type": "promotion_efficiency_signal",
                    "period": row["period"],
                    "store_id": row["store_id"],
                    "value": (
                        f"Store {row['store_id']} has low refund pressure but high subsidy "
                        f"intensity relative to revenue. Subsidy-to-revenue ratio is "
                        f"{pct(row['subsidy_to_revenue_ratio'])}."
                    ),
                    "decision_use": (
                        "Promotion efficiency should be checked before interpreting store "
                        "ranking or traffic performance as purely organic performance."
                    ),
                    "confidence": "medium",
                    "source": "retail_ops/sql/03_conservative_store_tags.sql",
                }
            )

    return facts


def write_report(derived_rows: list[dict[str, Any]], facts: list[dict[str, Any]]) -> None:
    total_stores = len(derived_rows)
    search_heavy_count = sum(
        1 for row in derived_rows if float(row["search_visit_share"]) >= 0.85
    )
    reference_store = max(derived_rows, key=lambda row: float(row["revenue"]))

    high_refund_stores = [
        row["store_id"]
        for row in derived_rows
        if float(row["refund_revenue_ratio"]) >= 0.10
        or float(row["refund_order_ratio"]) >= 0.12
    ]

    high_subsidy_low_refund_stores = [
        row["store_id"]
        for row in derived_rows
        if float(row["subsidy_to_revenue_ratio"]) >= 0.40
        and float(row["refund_revenue_ratio"]) < 0.10
    ]

    report = f"""# Cross-Store Meituan Metrics Comparison Report

## Purpose

Meituan merchant backend provides many detailed store-level indicators, but it is mainly designed for reviewing one store at a time.

For multi-store operations, the challenge is not the lack of data. The challenge is the difficulty of comparing many stores efficiently under the same metric structure.

This SQL-based extension reorganizes manually collected and anonymized Meituan backend metrics into a unified cross-store table. It calculates comparable indicators related to search traffic, conversion, refund pressure, promotion intensity, and visitor value.

## Data Scope

- Period: March 2026
- Stores: {total_stores} anonymized representative stores
- Regions: Qingdao urban area and Yantai urban area
- Store types: self-operated and partner-operated
- Category: contact lenses and care-solution related instant retail
- Source: manually organized Meituan merchant backend metrics

## Derived Metrics

The SQL layer calculates:

1. Search visit share = search visitors / store visitors
2. Search exposure-to-visit rate = search visitors / search exposure
3. Refund revenue ratio = refund amount / revenue
4. Refund order ratio = refund orders / valid orders
5. Promotion GMV to revenue ratio = promotion GMV / revenue
6. Merchant subsidy to revenue ratio = merchant subsidy / revenue
7. Merchant subsidy to promotion GMV ratio = merchant subsidy / promotion GMV
8. Revenue per visitor = revenue / store visitors

## Cross-Store Observations

Most stores in this sample are strongly search-driven. {search_heavy_count} out of {total_stores} stores have search visit share above 85%, suggesting that search visibility is a major traffic source in this category.

However, search traffic alone does not explain store performance. Store {reference_store['store_id']} has the highest revenue in this sample and can be reviewed as a reference store. It should not be copied directly without checking region, store type, product mix, promotion structure, refund pressure, fulfillment reliability, and customer profile.

Stores with visible refund pressure in the current sample: {", ".join(high_refund_stores) if high_refund_stores else "None"}.

Stores with high subsidy intensity but relatively low refund pressure: {", ".join(high_subsidy_low_refund_stores) if high_subsidy_low_refund_stores else "None"}.

These observations should not be treated as causal conclusions. They show why a cross-store SQL layer is useful before storing operational observations in an AI memory system.

## Generated Operational Memory Facts

The pipeline converts selected SQL observations into structured memory facts.

Current generated fact types include:

- `cross_store_pattern`
- `store_metric_signal`
- `risk_signal`
- `promotion_efficiency_signal`

Each fact includes scope, period, confidence, and source so that an AI assistant can reuse the observation conservatively.

## Decision-Support Principle

The memory layer should store operational observations with clear scope, time period, and confidence level.

It should avoid treating one store's pattern as a universal rule without checking region, store type, traffic source structure, refund pressure, promotion intensity, product mix, and visitor value.
"""

    (OUTPUT_DIR / "cross_store_comparison_report.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(":memory:")
    load_store_metrics(conn)

    derived_columns, derived_rows = run_sql_file(conn, "01_derived_metrics.sql")
    ranking_columns, ranking_rows = run_sql_file(conn, "02_cross_store_ranking.sql")
    tag_columns, tag_rows = run_sql_file(conn, "03_conservative_store_tags.sql")

    write_csv(OUTPUT_DIR / "derived_metrics_output.csv", derived_columns, derived_rows)
    write_csv(OUTPUT_DIR / "cross_store_ranking_output.csv", ranking_columns, ranking_rows)
    write_csv(OUTPUT_DIR / "store_tags_output.csv", tag_columns, tag_rows)

    facts = build_memory_facts(derived_rows)

    (OUTPUT_DIR / "generated_memory_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(derived_rows, facts)

    print("Retail pipeline completed.")
    print(f"Input: {INPUT_CSV}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
