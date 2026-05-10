# Retail Operations Extension

This folder extends the lifecycle-aware memory layer from livestream commerce facts to Meituan-style instant retail operations data.

The purpose is not to automatically label stores as good or bad. The purpose is to turn messy backend metrics into a limited, traceable, and verifiable decision-support path.

## Current Demo

### Demo 1: Store A Month-over-Month Diagnostic

Store A is analyzed across February, March, and April 2026, using natural calendar-month windows: `2026-02-01` to `2026-02-28`, `2026-03-01` to `2026-03-31`, and `2026-04-01` to `2026-04-30`.

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

## Implemented Retail Path

The current retail path is intentionally narrow:

```text
Meituan-style backend metrics
-> metric dictionary and lineage rules
-> offline SQL diagnostic
-> SQL-derived output tables
-> generated retail memory facts
-> data-contract validation
-> Store A retail retrieval evaluation
```

This does not yet represent a full multi-store decision-support system.

## Key Design Principle

The retail SQL layer is not designed to assign a fixed operating label from one threshold.

In this business context, Meituan instant retail stores compete through a chain of operating conditions:

```text
being seen -> being entered -> being ordered -> being selected again or maintaining market share
```

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are treated as operating levers inside this chain. They should not be interpreted as isolated causes by themselves.

Short-term ROI is not always the primary target. A new store may need subsidy to gain exposure and first orders. A store under external price pressure may need pricing or activity tools to defend visibility and market share. A store with enough traffic but weak conversion requires a different interpretation from a store with order growth but refund pressure.

The SQL layer therefore prepares comparison-ready diagnostic signals:

- visibility and entry profile;
- transaction and conversion profile;
- activity and subsidy profile;
- refund and invalid-order pressure;
- top-SKU evidence;
- month-over-month movement.

The current Store A demo is a single-store month-over-month diagnostic. It does not yet implement cross-store comparison or store-stage diagnosis.

The intended future work is to build a cross-store comparability gate and then use the memory layer to help judge store operating stage, comparable peer stores, and possible strategy choices.

## Data Contract and Metric Consistency

The retail demo uses `DATA_DICTIONARY.md` as the metric definition layer and `LINEAGE.md` as the claim-to-field lineage layer.

The validation script checks that:

- required canonical fields exist in the source and output files;
- forbidden alias fields are not reintroduced;
- generated memory facts reference known source fields;
- the generated retail memory fact file is valid JSON;
- source CSV, SQL output, metric dictionary, lineage, and generated facts remain consistent.

## Current Retail Memory Slots

The current generated retail memory facts use these canonical slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `order_quality_pressure_profile`
- `single_metric_attribution_guard`
- `top3_sku_product_mix_note`

Supporting SQL observations such as transaction recovery, order-conversion decline, average-order-value decline, refund-pressure improvement, and invalid-order-pressure improvement can support interpretation, but they are not standalone store-stage labels.

## Important Limitation

Promotion cycle dates are unknown in this demo.

Activity metrics are treated as operating-lever evidence, not as a clean intervention. The current Store A demo can support cautious interpretation, but it cannot support broad causal claims, cross-store generalization, or store-stage diagnosis.

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

## Future Work

The next demo should add a cross-store comparability gate.

That gate should check whether stores can be compared by:

- aligned reporting period;
- coarse market context;
- store type;
- order volume band;
- visibility and ranking profile;
- entry and order-conversion profile;
- activity-order share and subsidy profile;
- refund and invalid-order pressure;
- dominant top-SKU evidence;
- data completeness.

Only after this comparability check should the system generate cross-store operational interpretation or suggest whether a strategy from one store may transfer to another.
