# Retail Operations Analytical Checks

This file records the current analytical checks in the retail operations extension. The purpose is to make the prototype reviewable as a data-science decision-support project.

In this repository, "experiment" means a staged analytical check on whether selected Meituan backend metrics can be structured, traced, and discussed without losing metric definitions or evidence limits. It does not mean a randomized business experiment.

## Planned Threshold Review

The current Demo 2 thresholds used to create `comparison_limit_notes` are guardrail literals for this fixture-stage prototype. They are not estimated optimal cutoffs. When broader store coverage and repeated reporting windows are added, they should be reviewed with stability checks and simple sensitivity analysis before being used in a stronger pairwise comparability gate.

## E-01: Store A Month-over-Month Diagnostic

| Item | Content |
|---|---|
| Question | Can a single store's monthly operating movement be interpreted without reducing the result to one metric? |
| Input | Store A February, March, and April 2026 store-period metrics; Store A top-SKU evidence. |
| Transformation | `01_store_a_month_over_month_diagnostic.sql` derives month-over-month movement, ranking changes, traffic/conversion tradeoffs, refund pressure, invalid-order pressure, and top-SKU concentration evidence. |
| Output | `retail_ops/outputs/store_a_demo1_sql_output.csv`; generated Store A retail memory facts. |
| Pass condition | The system can describe exposure, entry, ranking, transaction, conversion, activity, refund, invalid-order, and top-SKU movement together. |
| Failure mode | Claiming that exposure, ranking, activity, conversion, refund pressure, or top-SKU movement alone caused the monthly result. |

## E-02: Demo 2 Same-Period Cross-Store Diagnostic

| Item | Content |
|---|---|
| Question | Can selected B-F store-period rows be structured before cross-store interpretation? |
| Input | `demo2_store_period_metrics.csv`; top search-term evidence; top-SKU transaction-amount evidence. |
| Transformation | `02_demo2_cross_store_comparability.sql` derives search-entry share/rate, activity-order share, refund pressure, invalid-order pressure, top-3 SKU concentration, `comparison_scope_flag`, and `comparison_limit_notes`. |
| Output | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; generated Demo 2 retail memory facts. |
| Pass condition | The system can discuss stores only within the documented same-period diagnostic scope and must preserve limits related to region context, store type, activity involvement, refund pressure, invalid-order pressure, product-mix evidence, and data completeness. |
| Failure mode | Ranking stores globally, treating same-period diagnostic readiness as pairwise comparability, treating `activity_cost_ratio_pct` as ROI, or transferring a promotion / price / SKU action without checking limits. |

## E-03: Retail Memory Fact Materialization

| Item | Content |
|---|---|
| Question | Can SQL diagnostic outputs be converted into retrieval-facing memory facts without losing source fields, observed values, source paths, and limitations? |
| Input | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/data/demo2_top_search_terms.csv`; `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`. |
| Transformation | `generate_demo2_retail_memory_facts.py` converts row-level diagnostics into slot-based retail memory facts. |
| Output | `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Pass condition | Each generated fact keeps the store entity, period, slot, observed values, calculation notes, source fields, primary source path, supporting source paths, lineage path, evidence-trace confidence, limitations, active status, and period granularity. |
| Failure mode | Mixing store-level and SKU-level fields, dropping source evidence, introducing undocumented fields, or letting top-search / top-SKU evidence appear without supporting source paths. |

## E-04: Unsupported Claim Guard

| Item | Content |
|---|---|
| Question | Does the system avoid overclaiming when the current evidence does not support a conclusion? |
| Input | Retail evaluation cases; Demo 2 generated facts; data dictionary; lineage rules. |
| Transformation | Scenario-based answer-behavior checks test whether retrieved evidence is used with the correct metric definitions and limitations. |
| Output | Retail evaluation result files under `eval/` and validation outputs under `retail_ops/outputs/`. |
| Pass condition | The system qualifies or refuses unsupported claims about causal attribution, audited profit, full 48-store generalization, final store ranking, promotion decisions, pairwise store comparability, or full product-category share. |
| Failure mode | Producing fluent but unsupported advice from isolated metrics, treating current Demo 2 as a completed comparability gate, or ignoring `comparison_limit_notes`. |

## Method Notes: What Demo 2 Guardrails Are Trying to Prevent

The Demo 2 thresholds are lightweight interpretation guardrails, not optimized business cutoffs. Their role is to make possible over-interpretation visible before SQL outputs are converted into memory facts or used in answer-boundary checks.

| Guardrail signal | Current trigger in SQL | Misreading it is meant to prevent |
|---|---|---|
| `comparison_scope_flag` | Period mismatch, missing required fields, or same-period diagnostic readiness. | Treating a row as usable when the reporting window or required evidence is incomplete. |
| `search_entry_share_pct` | `>= 85` -> `high_search_entry_dependence` | Treating strong search-entry dependence as overall store strength without checking entry quality, conversion, activity, refund, invalid-order, store type, and region context. |
| `activity_order_share_pct` | `>= 80` -> `high_activity_involvement`; `>= 65` -> `moderate_activity_involvement` | Treating promotion-supported order structure as normal baseline demand. |
| `refund_pressure_pct` | `>= 15` -> `high_refund_pressure`; `>= 10` -> `moderate_refund_pressure` | Reading transaction amount as clean demand when refund pressure may weaken the interpretation. |
| `invalid_order_pressure_pct` | `>= 12` -> `high_invalid_order_pressure`; `>= 8` -> `moderate_invalid_order_pressure` | Interpreting order volume without checking whether invalid-order pressure changes the operating picture. |
| `top3_sku_transaction_amount_share_pct` | `>= 25` -> `top3_sku_amount_concentration` | Treating a few high-value SKUs as if they described the full product-category structure. |
| `comparison_limit_notes` | Concatenated guardrail notes. | Letting a later answer ignore the limits already visible in the diagnostic row. |

## Fixture-Stage Threshold Sensitivity Snapshot

This table is a small review note, not a model-validation result. It checks how many of the five current Demo 2 rows would change a guardrail tag if the current threshold family moved by 5 percentage points.

| Guardrail family | Current threshold family | Shift down 5pp: changed rows | Shift up 5pp: changed rows | Interpretation |
|---|---:|---:|---:|---|
| Search-entry dependence | high `>= 85` | 0 / 5 | 4 / 5 | Current B-F rows sit close below 90, so the high-search tag is sensitive to a stricter threshold. |
| Activity involvement | high `>= 80`, moderate `>= 65` | 0 / 5 | 3 / 5 | Several rows sit around the current high/moderate boundary. This supports keeping the tag as a guardrail, not a transfer rule. |
| Refund pressure | high `>= 15`, moderate `>= 10` | 4 / 5 | 3 / 5 | Refund-pressure interpretation is threshold-sensitive in this small fixture. Broader repeated windows are needed before using it in a stronger gate. |
| Invalid-order pressure | high `>= 12`, moderate `>= 8` | 4 / 5 | 2 / 5 | Invalid-order tags are sensitive because several rows are near the current low-volume guardrail range. |
| Top-3 SKU amount concentration | concentration `>= 25` | 0 / 5 | 1 / 5 | Only one current row is above the concentration threshold; this remains lightweight product-mix evidence, not category-share analysis. |

These literal thresholds are deliberately simple at the current stage. A future comparability gate should test their stability and sensitivity across more store-period records before treating them as peer-group or strategy-transfer rules.
