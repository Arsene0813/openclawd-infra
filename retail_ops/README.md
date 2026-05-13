# Retail Operations Extension

This folder extends the lifecycle-aware memory layer from livestream commerce facts to Meituan-style instant retail operations data.

The purpose is to turn detailed single-store backend metrics into a limited, traceable, and verifiable decision-support path for cross-store comparison, because the Meituan merchant backend is rich but mainly designed for reviewing one store at a time rather than deciding which stores can be compared across a multi-store operation.

## Folder Scope

This folder contains the retail evidence layer:

| Component | Purpose |
|---|---|
| `data/` | Selected Meituan-style source tables and metric definitions |
| `sql/` | Diagnostic SQL for Demo 1 and Demo 2 |
| `outputs/` | Generated SQL outputs, validation results, and memory facts |
| `scripts/` | Local validation, generation, and loading scripts |
| `demo/` | Readable diagnostic write-ups |

The current retail path has two implemented demos and one planned future stage:

1. Demo 1: Store A month-over-month diagnosis.
2. Demo 2: same-period Stores B-F cross-store diagnostic with scope/limit checks.
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
- `eval/eval_retail_demo2_scope_boundary.py`
- `eval/results/eval_retail_demo2_scope_boundary_result.txt`
- `eval/eval_retail_demo2_answer_behavior.py`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

Demo 2 currently uses file-backed generated retail memory facts through a local prototype endpoint. It is not a production retail API endpoint.

## Implemented Retail Path

The current retail path is intentionally staged:

Meituan-style backend metrics -> DATA_DICTIONARY.md metric definitions -> canonical CSV fields -> SQL diagnostics -> SQL-derived output tables -> generated retail memory facts -> data-contract validation -> retrieval or offline evaluation -> cautious answer, qualified comparison, or refusal.

This now includes:

1. a Store A month-over-month diagnostic;
2. a same-period B-F cross-store diagnostic;
3. a future comparability gate after broader multi-store data is available.

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

The SQL layer therefore prepares evidence-bounded diagnostic signals:

- visibility and entry profile;
- transaction and conversion profile;
- activity and subsidy profile;
- refund and invalid-order pressure;
- top-SKU evidence;
- comparison-scope and limitation notes.

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

`retail_data_contract_validation_result.txt` captures the current lightweight data-contract guardrail for Demo 1 and Demo 2. Dedicated Demo 2 validators still cover staging data, SQL output, generated facts, and answer-boundary behavior.

## Future Comparability-Gate Design Notes

This folder includes three review documents for disciplined cross-store comparison:

- `retail_ops/FIELD_USAGE_REVIEW.md`
- `retail_ops/COMPARABILITY_GATE_V0.md`
- `retail_ops/EXPERIMENT_RESULTS.md`

Their role is to keep field usage, comparison rules, and limitation-preserving answers explicit before expanding toward broader store coverage.

`FIELD_USAGE_REVIEW.md` records the current decision not to rename existing fields before comparability-gate expansion.

`COMPARABILITY_GATE_V0.md` records candidate future gate factors and the current non-implementation boundary.

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

Demo 2 offline answer-boundary check checks whether comparison answers preserve metric definitions and limits. In particular, it checks that `activity_cost_ratio_pct` is not treated as ROI, top-SKU concentration is not treated as full product-category sales share, search-entry comparison stays tied to the correct fields, and promotion strategy transfer remains qualified.

## Current Boundary and Next Step

The current retail path should be read as a staged decision-support prototype, not as a complete multi-store operating system.

Promotion cycle dates are unknown in the current retail demos. Activity metrics are treated as operating-lever evidence, not as a clean intervention. Demo 1 and Demo 2 can support cautious interpretation, but they cannot support broad causal claims, full 48-store generalization, or store-stage diagnosis.

The next technical step is not to force a premature pairwise gate. The next step is to add broader multi-store data and repeated reporting windows before implementing a comparability gate.

The intended query shape is small:

- `reference_store_id`
- `candidate_store_id`
- `period_start`
- `period_end`
- `comparison_question_type`

The period fields are required because store comparability is not static. The same two stores may be more or less comparable under different promotion, refund, ranking, SKU, or competitive conditions.

The answer should return:

- supporting gap fields;
- refusal or qualification when evidence is not enough.

Any later expansion beyond the current B-F sample should keep the same discipline:

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

## Future Work: Comparability Gate

The current implemented retail scope stops at Demo 2.

A comparability gate is planned as future work. It should eventually help judge which stores can be compared, under what conditions, and what kind of operating action a comparison may support.

This is not currently implemented as a finished demo because the sample is still limited. Store comparability should depend on transaction order volume, transaction amount, whether the store is under activity or promotion, activity intensity, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows.

To avoid subjective regional classification, the current project treats `region_type` as weak context only. It is not a hard market-area classification or peer-store grouping rule.
