# Retail Comparability Experiment Review Cases

This file records review cases for the retail operations extension.

It is not a broad language-model benchmark. It is a focused behavior checklist for whether the SQL and memory-layer path preserves metric definitions, comparison limits, and unsupported-claim refusals.

## Current Evidence Scope

Current implemented retail scope:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: Stores B-F, same March 2026 reporting window.
- Demo 2 facts: generated file-backed retail memory facts.
- Current boundary: limited comparability diagnostic, not full 48-store automation.

## Review Cases

| Case | Question being tested | Evidence that should be used | Expected behavior | Failure mode to avoid |
|---:|---|---|---|---|
| 1 | Can two Demo 2 stores be compared at all? | period_start, period_end, comparison_scope_flag, comparison_limit_notes | Answer only inside the same-period diagnostic scope and preserve limitations. | Ranking stores as simply better or worse. |
| 2 | Can one store's activity strategy be copied to another store? | activity_orders, activity_order_share_pct, activity_cost, activity_cost_ratio_pct, merchant_subsidy_amount, platform_subsidy_amount, comparison_limit_notes | Qualify or refuse direct strategy transfer when promotion cycle and competitive-pressure evidence are missing. | Recommending subsidy increase or decrease directly from activity-cost ratio. |
| 3 | Does higher exposure explain better transaction performance? | exposure_users, store_average_rank, entry_users, transaction_amount, transaction_orders, order_conversion_rate_pct | Explain that exposure is part of the operating chain but not a standalone causal explanation. | Claiming exposure or ranking caused transaction growth by itself. |
| 4 | Is search traffic quality better in one store? | search_exposure_users, search_entry_users, search_entry_rate_pct, search_entry_share_pct, order_conversion_rate_pct | Discuss search-entry structure together with conversion. | Treating search-entry share alone as store quality. |
| 5 | Can order conversion be recomputed from valid orders? | order_users, entry_users, order_conversion_rate_pct, valid_orders | Preserve the backend definition: order_conversion_rate_pct = order_users / entry_users * 100. | Recomputing order conversion as valid_orders / entry_users. |
| 6 | Is estimated income equal to profit? | estimated_income_proxy, transaction_amount, activity_cost | State that estimated income is a platform-displayed proxy, not audited profit. | Making profit or margin claims from estimated_income_proxy. |
| 7 | Does top-3 SKU concentration prove a category strategy? | sku_name, sku_transaction_amount, top3_sku_transaction_amount_share_pct | Treat top-3 SKU evidence as lightweight product-mix evidence. | Calling it full product-category sales-share analysis. |
| 8 | Can missing top-SKU evidence be interpreted as zero concentration? | comparison_limit_notes, top-SKU source fields | Treat missing evidence as an interpretation limit. | Treating missing top-SKU data as zero. |
| 9 | Can a promotion-heavy store be compared with a low-activity store? | activity_order_share_pct, activity_cost_ratio_pct, comparison_limit_notes, transaction_amount, order_conversion_rate_pct | Compare only with activity and subsidy limits. | Saying one store is better without activity-context qualification. |
| 10 | Can refund pressure change the interpretation of transaction growth? | refund_amount, refund_pressure_pct, valid_orders, invalid_orders, invalid_order_pressure_pct, transaction_amount | Read transaction scale with refund and invalid-order pressure. | Treating more orders as automatically better. |
| 11 | Can a memory answer use one SQL observation as a standalone slot? | refund_pressure_improved, transaction_recovered_with_conversion_aov_tradeoff, canonical memory slots | Keep supporting SQL observations inside broader memory profiles. | Turning supporting booleans into standalone store-stage labels. |
| 12 | Can the system answer full 48-store strategy questions now? | Current Demo 1 and Demo 2 scope | Refuse or qualify full 48-store conclusions. | Pretending the current B-F sample is a complete 48-store system. |

## Acceptance Standard

A good answer should:

1. use canonical field names;
2. distinguish backend fields, SQL-derived diagnostics, and memory-facing slots;
3. preserve the Meituan backend definition of conversion metrics;
4. use comparison_limit_notes when a comparison is constrained;
5. avoid one-metric conclusions;
6. avoid direct subsidy, price, SKU, or ranking recommendations without sufficient context;
7. state when the current prototype does not yet support full 48-store conclusions.

## Current Interpretation

The project is strongest when it behaves like a disciplined decision-support prototype:

retrieve evidence -> check comparability -> preserve limits -> answer cautiously or refuse

It is weakest when it sounds like a generic AI operations dashboard.

The next technical improvement should therefore be a tested comparability gate, not more enterprise-style infrastructure.

## Retail Demo 2 Answer-Behavior Boundary Evaluation

Demo 2 now includes an offline answer-boundary check:

- `eval/eval_retail_demo2_answer_behavior.py`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

This check focuses on whether comparison answers preserve the implemented metric contract:

- `activity_cost_ratio_pct` is treated as activity-cost-ratio evidence, not ROI or profit margin.
- `top3_sku_transaction_amount_share_pct` is treated as lightweight top-SKU concentration evidence, not full product-category sales share.
- search-entry comparison stays tied to `search_entry_rate_pct`, `search_entry_share_pct`, `search_entry_users`, and `entry_users`.
- promotion or subsidy strategy transfer is qualified unless activity, subsidy, refund, invalid-order, and comparison-limit evidence support it.

This turns the current Demo 2 comparison boundary into a testable behavior.
