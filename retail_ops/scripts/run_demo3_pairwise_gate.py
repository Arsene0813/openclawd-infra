from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
SQL_PATH = ROOT / "retail_ops/sql/03_demo2_pairwise_comparability_gate.sql"
OUTPUT_PATH = ROOT / "retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv"

TABLE_NAME = "demo2_cross_store_comparability_output"


def load_csv_to_sqlite(conn: sqlite3.Connection, path: Path, table_name: str) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise SystemExit(f"Missing CSV header: {path}")

        quoted_fields = ", ".join(f'"{field}" TEXT' for field in fieldnames)

        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(f'CREATE TABLE "{table_name}" ({quoted_fields})')

        placeholders = ", ".join("?" for _ in fieldnames)
        quoted_columns = ", ".join(f'"{field}"' for field in fieldnames)
        insert_sql = f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})'

        rows = []
        for row in reader:
            rows.append([row[field] for field in fieldnames])

        conn.executemany(insert_sql, rows)


def main() -> int:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing input file: {INPUT_PATH}")

    if not SQL_PATH.exists():
        raise SystemExit(f"Missing SQL file: {SQL_PATH}")

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    load_csv_to_sqlite(conn, INPUT_PATH, TABLE_NAME)

    sql = SQL_PATH.read_text(encoding="utf-8")
    rows = conn.execute(sql).fetchall()

    if not rows:
        raise SystemExit("Demo 3 pairwise gate produced no rows.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys())

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            writer.writerow([row[header] for header in headers])

    print("Demo 3 pairwise comparability gate output generated.")
    print(f"Input: {INPUT_PATH}")
    print(f"SQL: {SQL_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Rows: {len(rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
