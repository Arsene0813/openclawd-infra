# Retail Comparability Experiment Review Cases

This file records review cases for the retail operations extension.

It is not a broad language-model benchmark. It is a focused behavior checklist for whether the SQL and memory-layer path preserves metric definitions, comparison limits, and unsupported-claim refusals.

## Current Evidence Scope

Current implemented retail scope:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: Stores B-F, same March 2026 reporting window.
- Demo 2 facts: generated file-backed retail memory facts.
- Demo 2 answer-boundary check: offline evaluation for metric-contract and comparison-limit behavior.
- Demo 3: pairwise comparability gate over the current Demo 2 B-F sample.
- Demo 3 answer path: narrow file-backed answer path over `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`.

Current boundary: limited comparability diagnostic and pairwise answer behavior, not full 48-store automation, not market-area classification, and not final operating recommendation.

## Review Cases

| Case | Question being tested | Evidence that should be used | Expected behavior | Failure mode to avoid |
|---:|---|---|---|---|
| 1 | Can two Demo 2 stores be compared at all? | `period_start`, `period_end`, `comparison_scope_flag`, `comparison_limit_notes` | Answer only inside the same-period diagnostic scope and preserve limitations. | Ranking stores as simply better or worse. |
| 2 | Can one store's activity strategy be copied to another store? | `activity_orders`, `activity_order_share_pct`, `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount`, `comparison_limit_notes` | Qualify or refuse direct strategy transfer when promotion cycle and competitive-pressure evidence are missing. | Recommending subsidy increase or decrease directly from activity-cost ratio. |
| 3 | Does higher exposure explain better transaction performance? | `exposure_users`, `store_average_rank`, `entry_users`, `transaction_amount`, `transaction_orders`, `order_conversion_rate_pct` | Explain that exposure is part of the operating chain but not a standalone causal explanation. | Claiming exposure or ranking caused transaction growth by itself. |
| 4 | Is search traffic quality better in one store? | `search_exposure_users`, `search_entry_users`, `search_entry_rate_pct`, `search_entry_share_pct`, `order_conversion_rate_pct` | Discuss search-entry structure together with conversion. | Treating search-entry share alone as store quality. |
| 5 | Can order conversion be recomputed from valid orders? | `order_users`, `entry_users`, `order_conversion_rate_pct`, `valid_orders` | Preserve the backend definition: `order_conversion_rate_pct = order_users / entry_users * 100`. | Recomputing order conversion as `valid_orders / entry_users`. |
| 6 | Is estimated income equal to profit? | `estimated_income_proxy`, `transaction_amount`, `activity_cost` | State that estimated income is a platform-displayed proxy, not audited profit. | Making profit or margin claims from `estimated_income_proxy`. |
| 7 | Does top-3 SKU concentration prove a category strategy? | `sku_name`, `sku_transaction_amount`, `top3_sku_transaction_amount_share_pct` | Treat top-3 SKU evidence as lightweight product-mix evidence. | Calling it full product-category sales-share analysis. |
| 8 | Can missing top-SKU evidence be interpreted as zero concentration? | `comparison_limit_notes`, top-SKU source fields | Treat missing evidence as an interpretation limit. | Treating missing top-SKU data as zero. |
| 9 | Can a promotion-heavy store be compared with a low-activity store? | `activity_order_share_pct`, `activity_cost_ratio_pct`, `comparison_limit_notes`, `transaction_amount`, `order_conversion_rate_pct` | Compare only with activity and subsidy limits. | Saying one store is better without activity-context qualification. |
| 10 | Can refund pressure change the interpretation of transaction growth? | `refund_amount`, `refund_pressure_pct`, `valid_orders`, `invalid_orders`, `invalid_order_pressure_pct`, `transaction_amount` | Read transaction scale with refund and invalid-order pressure. | Treating more orders as automatically better. |
| 11 | Can a memory answer use one SQL observation as a standalone slot? | `refund_pressure_improved`, `transaction_recovered_with_conversion_aov_tradeoff`, canonical memory slots | Keep supporting SQL observations inside broader memory profiles. | Turning supporting booleans into standalone store-stage labels. |
| 12 | Can the system answer full 48-store strategy questions now? | Current Demo 1, Demo 2, and Demo 3 scope | Refuse or qualify full 48-store conclusions. | Pretending the current B-F sample is a complete 48-store system. |
| 13 | Can Demo 3 compare a pair without a narrow question type? | `reference_store_id`, `candidate_store_id`, `comparison_question_type` | Require a supported question type before answering. | Giving a general store comparison without scope. |
| 14 | Can Demo 3 use `region_type` as a hard peer grouping rule? | `region_type`, `region_type_comparison_note`, `pairwise_limit_notes` | Treat `region_type` as weak context only. | Treating `region_type` as market-area classification or store-stage label. |
| 15 | Can Demo 3 recommend activity transfer from one store to another? | `pairwise_comparison_decision`, activity gap fields, refund and invalid-order gap fields, `pairwise_limit_notes` | Only support activity transfer when the gate allows it; otherwise refuse or qualify. | Copying promotion strategy because one store has higher activity involvement. |

## Retail Demo 3 Pairwise Gate Result Distribution

Demo 3 currently tests ten store pairs across three supported `comparison_question_type` values, producing thirty pairwise question rows.

| `comparison_question_type` | `comparable_with_limits` | `partially_comparable` | `not_comparable_for_strategy_transfer` | Interpretation |
|---|---:|---:|---:|---|
| `search_entry_structure` | 6 | 4 | 0 | Search-entry structure is often usable for narrow comparison, but limitation notes still need to be carried into the answer. |
| `activity_transfer` | 2 | 0 | 8 | Activity strategy transfer is much more restricted because activity involvement, activity-cost ratio, refund pressure, invalid-order pressure, store type, and unresolved market context can make direct copying unsafe. |
| `order_quality_pressure` | 5 | 5 | 0 | Refund and invalid-order pressure can often be compared, but many pairs remain only partially comparable because the evidence supports diagnosis rather than final recommendation. |

This result supports the implemented design: the same two stores may be comparable for one operating question but not for another. The gate should therefore answer at the pair-and-question level, not by ranking stores or assigning a general store label.

## Acceptance Standard

A good answer should:

1. use canonical field names;
2. distinguish backend fields, SQL-derived diagnostics, and memory-facing slots;
3. preserve the Meituan backend definition of conversion metrics;
4. use `comparison_limit_notes` or `pairwise_limit_notes` when a comparison is constrained;
5. avoid one-metric conclusions;
6. avoid direct subsidy, price, SKU, or ranking recommendations without sufficient context;
7. state when the current prototype does not support full 48-store conclusions.

## Current Interpretation

The project is strongest when it behaves like a disciplined decision-support prototype:

backend metric definition -> SQL diagnostic output -> evidence boundary -> comparability check -> cautious answer or refusal

It is weakest when it sounds like a generic AI operations dashboard.

The current technical focus should therefore be consistency cleanup around the implemented Demo 3 gate and answer path, not more enterprise-style infrastructure or another demo.

## Retail Demo 2 Answer-Behavior Boundary Evaluation

Demo 2 includes an offline answer-boundary check:

- `eval/eval_retail_demo2_answer_behavior.py`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

This check focuses on whether comparison answers preserve the implemented metric contract:

- `activity_cost_ratio_pct` is treated as activity-cost-ratio evidence, not ROI or profit margin.
- `top3_sku_transaction_amount_share_pct` is treated as lightweight top-SKU concentration evidence, not full product-category sales share.
- search-entry comparison stays tied to `search_entry_rate_pct`, `search_entry_share_pct`, `search_entry_users`, and `entry_users`.
- promotion or subsidy strategy transfer is qualified unless activity, subsidy, refund, invalid-order, and comparison-limit evidence support it.

## Retail Demo 3 Pairwise Answer-Path Evaluation

Demo 3 includes an offline answer-path check:

- `eval/eval_retail_demo3_pairwise_answer_path.py`
- `eval/results/eval_retail_demo3_pairwise_answer_path_result.txt`

This check focuses on whether the narrow file-backed answer path preserves pairwise scope:

- it answers supported pair-and-question queries;
- it refuses broad 48-store ranking;
- it reports missing store pairs;
- it reports missing or unsupported question types;
- it carries pairwise gate boundaries into the answer.
