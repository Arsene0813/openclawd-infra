# Retail Operations Extension

This folder contains the retail-operations evidence layer for a Meituan instant-retail decision-support prototype.

It starts from a practical multi-store problem: the Meituan merchant backend provides detailed single-store metrics, but it does not directly answer which store-period records can be compared, under what conditions, and what kind of operating judgment the comparison can support.

The single source of truth for field names and metric meanings is:

- `retail_ops/data/DATA_DICTIONARY.md`

## Folder Scope

| Component | Purpose |
|---|---|
| `data/` | Selected Meituan-style source tables, source notes, and metric definitions. |
| `sql/` | Diagnostic SQL for Demo 1 and Demo 2. |
| `outputs/` | Generated SQL outputs, validation results, and memory facts. |
| `scripts/` | Local validation, generation, and loading scripts. |
| `demo/` | Readable diagnostic write-ups. |

The current retail path has two implemented demos and one planned future stage:

1. Demo 1: Store A month-over-month diagnosis.
2. Demo 2: same-period Stores B-F cross-store diagnostic with scope and limit checks.
3. Future work: pairwise comparability-gate design after broader multi-store data is available.

## Suggested Retail Review Path

For a quick review of the retail extension, read these files in order:

1. `retail_ops/data/DATA_DICTIONARY.md`
2. `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`
3. `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`
4. `retail_ops/COMPARABILITY_GATE_V0.md`

This path shows the movement from metric definitions, to single-store diagnosis, to same-period cross-store diagnostic structure, and toward a future pairwise comparability-gate design.

## Current Demos

### Demo 1: Store A Month-over-Month Diagnostic

Store A is analyzed across February, March, and April 2026, using natural calendar-month windows:

- 2026-02-01 to 2026-02-28
- 2026-03-01 to 2026-03-31
- 2026-04-01 to 2026-04-30

The demo shows that store performance cannot be interpreted from one metric alone. Exposure, entry, ranking, transaction scale, order conversion, average order value, activity cost, refund pressure, invalid-order pressure, and top-SKU evidence can move in different directions.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

Core supporting files:

- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/LINEAGE.md`
- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/outputs/store_a_demo1_sql_output.csv`
- `retail_ops/outputs/store_a_demo1_interpretation_summary.csv`
- `retail_ops/outputs/generated_retail_memory_facts.json`

Validation and evaluation files:

- `retail_ops/scripts/validate_retail_data_contract.py`
- `retail_ops/outputs/retail_data_contract_validation_result.txt`
- `eval/eval_retail.py`
- `eval/eval_retail_cases.json`
- `eval/eval_retail_report.md`

### Demo 2: Same-Period Cross-Store Diagnostic

Demo 2 extends the retail path from a single-store month-over-month diagnostic to a same-period cross-store diagnostic.

It uses five anonymized stores, B-F, all from the same reporting window:

- 2026-03-01 to 2026-03-31

Demo 2 structures selected backend metrics under the same reporting window and field contract, derives cautious diagnostic signals, and preserves interpretation limits before any operating recommendation is made.

Current Demo 2 scope:

- same-period diagnostic structure;
- selected store-period metrics;
- selected top search-term evidence;
- selected top-SKU evidence;
- SQL-derived diagnostic fields;
- generated retail memory facts;
- evaluation checks for metric-boundary preservation.

Current Demo 2 boundary:

- not an implemented pairwise comparison decision system;
- not a store ranking;
- not a strategy-transfer approval system;
- not a full 48-store automated operating platform.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

Core supporting files:

- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_source_notes.md`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`
- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

## Script Notes

| Script | Meaning |
|---|---|
| `retail_ops/scripts/validate_demo2_staging_data.py` | Validates Demo 2 source-table structure. |
| `retail_ops/scripts/validate_demo2_comparability_output.py` | Validates the Demo 2 diagnostic output contract, not a pairwise gate. |
| `retail_ops/scripts/generate_demo2_retail_memory_facts.py` | Converts Demo 2 diagnostic output into generated retail memory facts. |
| `retail_ops/scripts/validate_demo2_retail_memory_facts.py` | Validates generated Demo 2 retail memory fact structure. |
| `retail_ops/scripts/validate_retail_data_contract.py` | Checks retail field-contract consistency across dictionary, source, output, and facts. |

## Implemented Retail Path

The implemented path is:

```text
selected Meituan backend metrics
-> canonical field dictionary
-> SQL diagnostic output
-> generated memory facts
-> scenario-based evaluation
-> boundary-preserving retrieval answer
```

The project deliberately keeps source fields, observed values, calculation notes, source paths, supporting source paths, confidence labels, and limitations with the facts.

This is what prevents a later answer from treating one metric as a complete operating conclusion.

## Future Work: Pairwise Comparability Gate

The next planned stage is a pairwise comparability gate.

It should judge whether two selected store-period records can be compared for a specific operating question.

The gate should consider:

- order volume;
- transaction amount;
- activity involvement and activity intensity;
- explicit activity status or campaign-calendar evidence if available;
- store type;
- region and market context;
- competition environment;
- SKU structure;
- refund pressure;
- invalid-order pressure;
- repeated reporting windows.

The current `region_type` field remains weak context only. It should not be used as a hard market-area classification or peer-grouping rule without broader store data and stronger local market evidence.

Future gate design note:

- `retail_ops/COMPARABILITY_GATE_V0.md`
