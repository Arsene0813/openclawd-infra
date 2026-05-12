# Retail Operations Extension

This folder extends the lifecycle-aware memory layer from livestream commerce facts to Meituan-style instant retail operations data.

The purpose is to turn messy single-store backend metrics into a limited, traceable, and verifiable decision-support path for cross-store comparison.

## Folder Scope

This folder contains the retail evidence layer:

| Component | Purpose |
|---|---|
| `data/` | Selected Meituan-style source tables and metric definitions |
| `sql/` | Diagnostic SQL for Demo 1, Demo 2, and Demo 3 |
| `outputs/` | Generated SQL outputs, validation results, and memory facts |
| `scripts/` | Local validation, generation, and loading scripts |
| `demo/` | Readable diagnostic write-ups |

The current retail path has three fixed demos:

1. Demo 1: Store A month-over-month diagnosis.
2. Demo 2: same-period Stores B-F comparability diagnostic.
3. Demo 3: pairwise comparability gate over the current Demo 2 B-F sample.

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

### Demo 2: Same-Period Cross-Store Comparability Diagnostic

Demo 2 extends the retail path from a single-store month-over-month diagnostic to a same-period cross-store diagnostic.

It uses five anonymized stores, B-F, all from the same reporting window:

- 2026-03-01 to 2026-03-31

Demo 2 structures comparable backend metrics, derives cautious diagnostic signals, and preserves interpretation limits before any operating recommendation is made.

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
- `eval/eval_retail_demo2_comparability_gate.py`
- `eval/results/eval_retail_demo2_comparability_gate_result.txt`
- `eval/eval_retail_demo2_answer_behavior.py`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

The Demo 2 API endpoint is:

- `/chat_retail_ops_demo2_kb`

This endpoint is file-backed and uses generated Demo 2 retail memory facts. It is separate from `/chat_retail_ops_kb`, which remains the Store A Demo 1 endpoint.

### Demo 3: Pairwise Comparability Gate

Demo 3 turns the Demo 2 B-F same-period diagnostic into a pairwise comparability gate.

It compares every pair of stores for three narrow question types:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

The output is:

- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`

Core supporting files:

- `retail_ops/sql/03_demo2_pairwise_comparability_gate.sql`
- `retail_ops/scripts/run_demo3_pairwise_gate.py`
- `retail_ops/scripts/validate_demo3_pairwise_gate_output.py`
- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`
- `retail_ops/demo/demo_3_pairwise_comparability_gate.md`
- `retail_ops/scripts/answer_demo3_pairwise_gate.py`
- `retail_ops/demo/demo_3_pairwise_answer_path.md`
- `eval/eval_retail_demo3_pairwise_gate.py`
- `eval/results/eval_retail_demo3_pairwise_gate_result.txt`
- `eval/eval_retail_demo3_pairwise_answer_path.py`
- `eval/results/eval_retail_demo3_pairwise_answer_path_result.txt`

Demo 3 keeps `region_type` only as weak context through `region_type_comparison_note`. The purpose is to make comparability testable before any store comparison or operating strategy transfer is attempted.

Demo 3 is currently implemented as SQL output, saved CSV output, documentation, validation, offline evaluation, and a narrow file-backed answer path.

## Implemented Retail Path

The current retail path is intentionally staged:

Meituan-style backend metrics -> DATA_DICTIONARY.md metric definitions -> canonical CSV fields -> SQL diagnostics -> SQL-derived output tables -> generated retail memory facts -> data-contract validation -> retrieval or offline evaluation -> cautious answer, qualified comparison, or refusal.

This now includes:

1. a Store A month-over-month diagnostic;
2. a same-period B-F cross-store diagnostic;
3. a B-F pairwise comparability gate for narrow operating questions.

## Readable Architecture

Meituan backend screenshots or exports are organized into canonical CSV fields based on `DATA_DICTIONARY.md`.

SQL diagnostics then create ratios, shares, pressure indicators, and comparison-scope notes.

Memory facts preserve store-period evidence, observed values, confidence, and limitations.

Retrieval and evaluation check whether later answers stay inside the supported evidence boundary.

## Key Design Principle

Meituan instant-retail stores compete through a chain of operating conditions:

being seen -> being entered -> being ordered -> being selected again or maintaining market share.

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are treated as operating levers inside this chain.

Short-term activity-cost efficiency is not always the primary target. A new store may need subsidy to gain exposure and first orders. A store under external price pressure may need pricing or activity tools to defend visibility and market share. A store with enough traffic but weak conversion requires a different interpretation from a store with order growth but refund pressure.

The SQL layer therefore prepares comparison-ready diagnostic signals:

- visibility and entry profile;
- transaction and conversion profile;
- activity and subsidy profile;
- refund and invalid-order pressure;
- top-SKU evidence;
- comparison-scope and limitation notes.

## Data Contract and Metric Consistency

The retail demo uses `DATA_DICTIONARY.md` as the metric definition layer and `LINEAGE.md` as the claim-to-field lineage layer.

The validation script checks that:

- required canonical fields exist in the source and output files;
- forbidden alias fields are not reintroduced;
- generated memory facts reference known source fields;
- the generated retail memory fact file is valid JSON;
- source CSV, SQL output, metric dictionary, lineage, and generated facts remain consistent.

Main validation file:

- `retail_ops/scripts/validate_retail_data_contract.py`

Saved validation output:

- `retail_ops/outputs/retail_data_contract_validation_result.txt`

`retail_data_contract_validation_result.txt` validates the original Store A / Demo 1 retail data contract. Demo 2 and Demo 3 use additional dedicated validators and evaluation files for cross-store comparability and pairwise gate behavior.

## Comparability-Gate Documentation

This folder includes three review documents for disciplined cross-store comparison:

- `retail_ops/FIELD_USAGE_REVIEW.md`
- `retail_ops/COMPARABILITY_GATE_V0.md`
- `retail_ops/EXPERIMENT_RESULTS.md`

Their role is to keep field usage, comparison rules, and limitation-preserving answers explicit before expanding toward broader store coverage.

`FIELD_USAGE_REVIEW.md` records that this patch does not rename existing fields.

`COMPARABILITY_GATE_V0.md` defines the first review version of the comparability gate.

`EXPERIMENT_RESULTS.md` records review cases for limitation-preserving answers.

## Current Retail Memory Slots

The current generated retail memory facts use these canonical slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `single_metric_attribution_guard`
- `top3_sku_product_mix_note`

Supporting SQL observations such as transaction recovery, order-conversion decline, average-order-value decline, refund-pressure improvement, invalid-order-pressure improvement, activity-order share, and activity-cost ratio can support interpretation, but they are not standalone store-stage labels.

## Current Evaluation Boundary

The Store A retail retrieval evaluation checks whether the system can:

- retrieve visibility and entry profile evidence;
- retrieve activity-lever evidence;
- describe transaction and conversion tradeoffs;
- describe refund and invalid-order pressure;
- warn against single-metric attribution;
- limit top-SKU interpretation;
- refuse unknown-store stage claims;
- refuse unsupported cross-store strategy claims.

Demo 2 offline facts evaluation checks whether generated B-F facts cover visibility, activity, transaction/conversion, order-quality pressure, top-SKU evidence, and attribution-guard slots.

Demo 2 answer-behavior boundary evaluation checks whether comparison answers preserve metric definitions and limits. In particular, it checks that `activity_cost_ratio_pct` is not treated as ROI, top-SKU concentration is not treated as full product-category sales share, search-entry comparison stays tied to the correct fields, and promotion strategy transfer remains qualified.

Demo 3 pairwise gate evaluation checks whether pairwise output preserves three narrow question types, keeps `region_type` as weak context, avoids best-store ranking, and documents the new pairwise fields.

Demo 3 answer-path evaluation checks whether the file-backed answer path answers supported pairwise questions, includes limitation notes, reports missing pairs or missing question types, and refuses full 48-store ranking.

## Current Boundary and Next Step

The current retail path should be read as a staged decision-support prototype, not as a complete multi-store operating system.

Promotion cycle dates are unknown in the current retail demos. Activity metrics are treated as operating-lever evidence, not as a clean intervention. Demo 1, Demo 2, and Demo 3 can support cautious interpretation, but they cannot support broad causal claims, full 48-store generalization, or store-stage diagnosis.

The next technical step is not to create a new demo. The next step is to decide whether the existing Demo 3 file-backed answer path should remain deterministic or be exposed through a narrow retrieval/API layer.

The intended query shape is small:

- `reference_store_id`
- `candidate_store_id`
- `comparison_question_type`

The answer should return:

- `pairwise_comparison_decision`
- `pairwise_limit_notes`
- supporting gap fields;
- refusal or qualification when evidence is not enough.

Expansion beyond the current B-F sample should keep the same discipline:

- aligned reporting period;
- coarse market context;
- store type;
- order volume band;
- visibility and ranking profile;
- entry and order-conversion profile;
- activity-order share and subsidy profile;
- refund and invalid-order pressure;
- dominant top-SKU evidence;
- data completeness;
- explicit limitation notes.
