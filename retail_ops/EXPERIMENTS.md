# Retail Operations Experiment Map

This file records the current analytical experiments in the retail operations extension. The purpose is to make the prototype easier to review as a data-science decision-support project.

These experiments are not production experiments and not causal A/B tests. They are staged checks for whether selected Meituan backend metrics can be structured, traced, and discussed without losing metric definitions or evidence limits.

## Experiment 1: Store A Month-over-Month Diagnostic

| Item | Content |
|---|---|
| Question | Can a single store's monthly operating movement be interpreted without reducing the result to one metric? |
| Input | Store A February, March, and April 2026 store-period metrics; Store A top-SKU evidence. |
| Transformation | `01_store_a_month_over_month_diagnostic.sql` derives month-over-month movement, ranking changes, traffic/conversion tradeoffs, refund pressure, invalid-order pressure, and top-SKU concentration evidence. |
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
| Failure mode | Ranking stores globally, treating same-period diagnostic readiness as pairwise comparability, treating `activity_cost_ratio_pct` as ROI, or transferring a promotion / price / SKU action without checking limits. |

## Experiment 3: Retail Memory Fact Generation

| Item | Content |
|---|---|
| Question | Can SQL diagnostic outputs be converted into retrieval-facing memory facts without losing source fields, observed values, source paths, and limitations? |
| Input | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/data/demo2_top_search_terms.csv`; `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`. |
| Transformation | `generate_demo2_retail_memory_facts.py` converts row-level diagnostics into slot-based retail memory facts. |
| Output | `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Pass condition | Each generated fact keeps the store entity, period, slot, observed values, calculation notes, source fields, primary source path, supporting source paths, lineage path, confidence, limitations, active status, and period granularity. |
| Failure mode | Mixing store-level and SKU-level fields, dropping source evidence, introducing undocumented fields, or letting top-search / top-SKU evidence appear without supporting source paths. |

## Experiment 4: Unsupported Claim Guard

| Item | Content |
|---|---|
| Question | Does the system avoid overclaiming when the current evidence does not support a conclusion? |
| Input | Retail evaluation cases; Demo 2 generated facts; data dictionary; lineage rules. |
| Transformation | Scenario-based answer-behavior checks test whether retrieved evidence is used with the correct metric definitions and limitations. |
| Output | Retail evaluation result files under `eval/` and validation outputs under `retail_ops/outputs/`. |
| Pass condition | The system qualifies or refuses unsupported claims about causal attribution, audited profit, full 48-store generalization, final store ranking, promotion decisions, pairwise store comparability, or full product-category share. |
| Failure mode | Producing fluent but unsupported advice from isolated metrics, treating current Demo 2 as a completed comparability gate, or ignoring `comparison_limit_notes`. |
