# Retail Operations Extension

This folder contains the retail-operations evidence layer for a Meituan instant-retail decision-support prototype.

It starts from a practical multi-store problem: the Meituan merchant backend provides detailed single-store metrics, but it does not directly answer which store-period records can be compared, under what conditions, and what kind of operating judgment the comparison can support.

## Folder Scope

This folder contains the retail evidence layer.

| Component | Purpose |
|---|---|
| `data/` | Selected Meituan-style source tables and metric definitions. |
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

### Demo 2: Same-Period Cross-Store Diagnostic (B-F)

Demo 2 extends the retail path from a single-store month-over-month diagnostic to a same-period cross-store diagnostic.

It uses five anonymized stores, B-F, all from the same reporting window:

- 2026-03-01 to 2026-03-31

Demo 2 structures selected backend metrics under the same reporting window and field contract, derives cautious diagnostic signals, and preserves interpretation limits before any operating recommendation is made.

Core supporting files:

- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_source_notes.md`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`
- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

Validation and evaluation files:

- `retail_ops/scripts/validate_demo2_staging_data.py`
- `retail_ops/scripts/validate_demo2_comparability_output.py`
- `retail_ops/scripts/generate_demo2_retail_memory_facts.py`
- `retail_ops/scripts/validate_demo2_retail_memory_facts.py`
- `eval/eval_retail_demo2_facts.py`
- `eval/results/eval_retail_demo2_facts_result.txt`
- `eval/eval_retail_demo2_scope_boundary.py`
- `eval/results/eval_retail_demo2_scope_boundary_result.txt`
- `eval/eval_retail_demo2_answer_behavior.py`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

Demo 2 currently uses file-backed generated retail memory facts through a local prototype endpoint. It is not a production Meituan API integration.

## Implemented Retail Path

The current retail path is intentionally staged:

    Meituan-style backend metrics
    -> DATA_DICTIONARY.md metric definitions
    -> canonical CSV fields
    -> SQL diagnostics
    -> SQL-derived output tables
    -> generated retail memory facts
    -> data-contract validation
    -> retrieval or offline evaluation
    -> cautious answer, qualified comparison, or refusal

This now includes:

1. a Store A month-over-month diagnostic;
2. a same-period B-F cross-store diagnostic;
3. a future comparability gate after broader multi-store data is available.

## Readable Architecture

Meituan backend screenshots or exports are organized into canonical CSV fields based on `DATA_DICTIONARY.md`. SQL diagnostics then create ratios, shares, pressure indicators, and comparison-scope notes.

Memory facts preserve store-period evidence, observed values, confidence, and limitations. Retrieval and evaluation check whether later answers stay inside the supported evidence boundary.

## Key Design Principle

Meituan instant-retail stores compete through a chain of operating conditions:

    being seen -> being entered -> being ordered -> being selected again or maintaining share

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are treated as operating levers inside this chain.

Short-term activity-cost efficiency is not always the primary target. A new store may need subsidy to gain exposure and first orders. A store under external price pressure may need pricing or activity tools to defend visibility and market share. A store with enough traffic but weak conversion requires a different interpretation from a store with order growth but refund pressure.

## Data Contract and Metric Consistency

The retail demo uses `DATA_DICTIONARY.md` as the metric definition layer and `LINEAGE.md` as the claim-to-field lineage layer.

The validation script checks that:

- required current-scope files exist;
- canonical dictionary boundary phrases are preserved;
- Demo 1 and Demo 2 outputs expose required headers;
- forbidden alias fields are not reintroduced;
- generated memory fact files keep expected structure, source paths, and known source fields.

Main validation file:

- `retail_ops/scripts/validate_retail_data_contract.py`

Saved validation output:

- `retail_ops/outputs/retail_data_contract_validation_result.txt`

## Current Boundary and Future Comparability Gate

The current retail path is a staged decision-support prototype. Demo 1 and Demo 2 support cautious interpretation, but they do not support broad causal claims, full 48-store generalization, or store-stage diagnosis.

Retail Demo 2 is the current implemented retail endpoint. It is a same-period B-F cross-store diagnostic with generated retail memory facts, scope checks, and limitation notes. A pairwise comparability gate is future work.

The next technical step is to expand store-period coverage and repeated reporting windows before implementing that gate. A reliable gate should judge whether selected store-period records can be compared for a specific operating question, using transaction order volume, transaction amount, current activity involvement and intensity based on existing activity fields, explicit activity status or campaign-calendar evidence if available, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

To avoid subjective regional classification, the current project treats `region_type` as weak context only. It is not a hard market-area classification or peer-store grouping rule.
