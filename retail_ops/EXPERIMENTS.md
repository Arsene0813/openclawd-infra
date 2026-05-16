# Retail Operations Experiment Map

This file records the current analytical experiments in the retail operations extension.

In this repository, "experiment" means a staged analytical check on whether the data path preserves metric definitions and answer boundaries.

It does not mean a randomized business experiment or causal A/B test.

## Document Responsibility

This file owns implementation checks and evaluation-oriented experiment descriptions.

It should record the question, evidence path, expected behavior, current result, pass condition, and failure mode for each staged check. It should not repeat the full business narrative or the full future pairwise-gate design.

## Current Experiment Scope

The current experiments test whether selected Meituan backend metrics can be:

1. structured into canonical fields;
2. processed through SQL diagnostics;
3. converted into retrieval-facing memory facts;
4. discussed without losing metric definitions or evidence limits.

The current experiments do not prove causal business effects.

## Planned Threshold Review

The current Demo 2 thresholds used to create `comparison_limit_notes` are fixture-stage literal thresholds for this prototype.

They are not estimated optimal cutoffs.

When broader store coverage and repeated reporting windows are added, these thresholds should be reviewed with stability checks and simple sensitivity analysis before being used in a stronger pairwise comparability gate.

## Experiment 1: Store A Month-over-Month Diagnostic

| Item | Content |
|---|---|
| Question | Can a single store's monthly operating movement be interpreted without reducing the result to one metric? |
| Input | Store A February, March, and April 2026 store-period metrics; Store A top-SKU evidence. |
| Transformation | `01_store_a_month_over_month_diagnostic.sql` derives month-over-month movement, ranking changes, traffic and conversion tradeoffs, refund pressure, invalid-order pressure, and top-SKU concentration evidence. |
| Output | `retail_ops/outputs/store_a_demo1_sql_output.csv`; generated Store A retail memory facts. |
| Pass condition | The system can describe exposure, entry, ranking, transaction, conversion, activity, refund, invalid-order, and top-SKU movement together. |
| Failure mode | Claiming that exposure, ranking, activity, conversion, refund pressure, or top-SKU movement alone caused the monthly result. |

## Experiment 2: Demo 2 Same-Period Cross-Store Diagnostic

| Item | Content |
|---|---|
| Question | Can selected B-F store-period rows be structured before cross-store interpretation? |
| Input | `demo2_store_period_metrics.csv`; top search-term evidence; top-SKU transaction-amount evidence. |
| Transformation | `02_demo2_cross_store_comparability.sql` derives search-entry share/rate, activity-order share, refund pressure, invalid-order pressure, top-3 SKU concentration, `comparison_scope_flag`, and `comparison_limit_notes`. |
| Output | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; generated Demo 2 retail memory facts. |
| Pass condition | The system can discuss stores only within the documented same-period diagnostic scope and must preserve limits related to region context, store type, activity involvement, refund pressure, invalid-order pressure, product-mix evidence, and data completeness. |
| Failure mode | Ranking stores globally, treating same-period diagnostic readiness as pairwise comparability, treating `activity_cost_ratio_pct` as ROI, or transferring a promotion, price, or SKU action without checking limits. |

## Experiment 3: Retail Memory Fact Generation

| Item | Content |
|---|---|
| Question | Can SQL diagnostic outputs be converted into retrieval-facing memory facts without losing source fields, observed values, source paths, and limitations? |
| Input | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/data/demo2_top_search_terms.csv`; `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`. |
| Transformation | `generate_demo2_retail_memory_facts.py` converts row-level diagnostics into slot-based retail memory facts. |
| Output | `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Pass condition | Each generated fact keeps the store entity, period, slot, observed values, calculation notes, source fields, primary source path, supporting source paths, lineage path, confidence, limitations, active status, and period granularity. |
| Failure mode | Mixing store-level and SKU-level fields, dropping source evidence, introducing undocumented fields, or letting top-search or top-SKU evidence appear without supporting source paths. |

## Experiment 4: Unsupported Claim Guard

| Item | Content |
|---|---|
| Question | Does the system avoid overclaiming when the current evidence does not support a conclusion? |
| Input | Retail evaluation cases; Demo 2 generated facts; data dictionary; lineage rules. |
| Transformation | Scenario-based answer-behavior checks test whether retrieved evidence is used with the correct metric definitions and limitations. |
| Output | Retail evaluation result files under `eval/` and validation outputs under `retail_ops/outputs/`. |
| Pass condition | The system qualifies or refuses unsupported claims about causal attribution, audited profit, full 48-store generalization, final store ranking, promotion decisions, pairwise store comparability, or full product-category share. |
| Failure mode | Producing fluent but unsupported advice from isolated metrics, treating current Demo 2 as a completed pairwise decision system, or ignoring `comparison_limit_notes`. |

## Future Experiment: Pairwise Comparability Gate Contract Stub

| Item | Content |
|---|---|
| Question | Can the future pairwise comparability gate contract be frozen before implementation? |
| Input | `retail_ops/COMPARABILITY_GATE_V0.md`. |
| Transformation | A lightweight eval stub checks that the planned input triple, output enum, and blocking-factor list are documented. |
| Output | `eval/eval_future_comparability_gate_contract.py`. |
| Pass condition | The stub confirms the future contract exists while not claiming that the pairwise gate is implemented. |
| Failure mode | Treating Demo 2 row-level diagnostic readiness as a pairwise comparability decision. |

## Method Notes: What Demo 2 Guardrails Are Trying to Prevent

The Demo 2 thresholds are lightweight interpretation guardrails, not optimized business cutoffs.

Their role is to make possible over-interpretation visible before SQL outputs are converted into memory facts or used in answer-boundary checks.

| Guardrail signal | Current trigger in SQL | Misreading it is meant to prevent |
|---|---|---|
| `comparison_scope_flag` | Period mismatch, missing required fields, or same-period diagnostic readiness | Treating a row as usable when the reporting window or required evidence is incomplete. |
| `search_entry_share_pct` | `>= 85` means `high_search_entry_dependence` | Treating strong search-entry dependence as overall store strength without checking entry quality, conversion, activity, refund, invalid-order, store type, and region context. |
| `activity_order_share_pct` | `>= 80` means `high_activity_involvement`; `>= 65` means `moderate_activity_involvement` | Treating promotion-supported order structure as normal baseline demand. |
| `refund_pressure_pct` | `>= 15` means `high_refund_pressure`; `>= 10` means `moderate_refund_pressure` | Reading transaction amount as clean demand when refund pressure may weaken the interpretation. |
| `invalid_order_pressure_pct` | `>= 12` means `high_invalid_order_pressure`; `>= 8` means `moderate_invalid_order_pressure` | Interpreting order volume without checking whether invalid-order pressure changes the operating picture. |
| `top3_sku_transaction_amount_share_pct` | `>= 25` means `top3_sku_amount_concentration` | Treating a few high-value SKUs as if they described the full product-category structure. |
| `comparison_limit_notes` | Concatenated guardrail notes | Letting a later answer ignore the limits already visible in the diagnostic row. |

These literal thresholds are deliberately simple at the current stage.

A future comparability gate should test their stability and sensitivity across more store-period records before treating them as peer-group or strategy-transfer rules.
