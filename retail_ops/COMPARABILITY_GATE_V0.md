# Comparability Gate V0

This document defines the first review version of the cross-store comparability gate for the retail operations extension.

The gate exists because Meituan merchant-backend data is rich but mainly single-store-oriented. For a multi-store instant-retail business, the harder question is not whether one store improved. The harder question is whether two store-period rows can be compared before any operating interpretation is made.

This file is design-level documentation. It does not rename existing fields. It does not add new SQL output columns. It does not add new memory slots.

## Current Scope

Current implemented scope:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: Stores B-F, same March 2026 reporting window.
- Current Demo 2 output: limited same-period cross-store comparability diagnostic.
- Current system boundary: not a full 48-store decision-support system.

## Operating Chain

The retail operating chain used by this project is:

being seen -> being entered -> being ordered -> being selected again or maintaining market share

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are operating levers inside this chain. They should be read through store context, not treated as isolated goals.

## Gate Question

Before comparing two stores, the system should ask:

Are these store-period rows comparable enough for the specific question being asked?

The answer can be different for different questions.

For example:

- Two stores may be comparable for search-entry structure.
- The same two stores may not be comparable for subsidy strategy.
- A store may be comparable within the same month but not across a different activity cycle.
- A store with top-SKU evidence missing should not be compared on product-mix concentration.

## Gate Dimensions

| Gate dimension | Existing evidence fields | Why it matters | Current boundary |
|---|---|---|---|
| Reporting period alignment | period_start, period_end, period_month, comparison_scope_flag | Different periods may reflect different activity cycles, demand conditions, or competition. | Demo 2 currently uses March 2026 same-period rows. |
| Region and market context | region_type | Market context affects demand, competition, delivery radius, purchasing power, and customer behavior. | region_type is coarse and not a full city/county/community classification. |
| Store operating format | store_type | Self-operated, warehouse-style, and partner-store contexts may not be directly comparable. | Current field is metadata; it is not a performance label. |
| Visibility and ranking profile | exposure_users, exposure_times, store_average_rank, search_exposure_users, search_average_rank | A store must first be seen before entry or order signals can be interpreted. | Ranking and exposure do not prove transaction causality by themselves. |
| Entry structure | entry_users, entry_times, search_entry_users, search_entry_rate_pct, search_entry_share_pct | Search-heavy traffic may behave differently from other traffic sources. | Source-level exposure and entry signals should not be over-summed or over-attributed. |
| Transaction and conversion profile | transaction_amount, transaction_orders, order_users, order_conversion_rate_pct, payment_users, payment_conversion_rate_pct, average_order_value | More orders, higher transaction amount, and stronger conversion are not the same signal. | order_conversion_rate_pct follows the backend definition and must not be recomputed from valid_orders. |
| Activity and subsidy profile | activity_original_transaction_amount, activity_orders, activity_cost, merchant_subsidy_amount, platform_subsidy_amount, activity_cost_ratio_pct, activity_order_share_pct | Activity may support exposure, new-store ramp-up, or market-share defense. | Activity data alone does not prove a clean intervention effect. |
| Refund and invalid-order pressure | refund_amount, full_refund_orders, refund_orders_all_or_partial, valid_orders, invalid_orders, refund_pressure_pct, invalid_order_pressure_pct | Order growth with refund or cancellation pressure should be interpreted differently from clean order growth. | Refund amount is dated by refund-success date and is not a perfect original-order cohort measure. |
| Top-SKU evidence | sku_name, sku_transaction_amount, sales_volume, top3_sku_transaction_amount_share_pct | Concentrated leading SKUs may affect transferability of operating signals. | Top-3 SKU evidence is lightweight product-mix evidence, not full product-category sales share. |
| Missing evidence | comparison_limit_notes | Missing data should create limits instead of being silently treated as zero. | Missing top-SKU evidence, missing transaction amount, or missing valid orders should constrain interpretation. |

## Implemented SQL Scope Flags

The current Demo 2 SQL output uses `comparison_scope_flag` as an implemented SQL-derived scope/readiness field. These are current implemented values, not future conceptual labels.

| `comparison_scope_flag` value | Meaning | Allowed interpretation | Not allowed |
|---|---|---|---|
| `same_period_diagnostic_ready` | The row is inside the current Demo 2 same-period diagnostic scope. | Discuss same-period diagnostic evidence with limits. | Treat as a performance ranking or final operating recommendation. |
| `not_comparable_period_mismatch` | The row does not match the Demo 2 reporting window. | Refuse or qualify same-period comparison. | Mix periods casually as if demand, activity, and competition were aligned. |
| `insufficient_data` | One or more required diagnostic fields are missing. | State the missing evidence and constrain interpretation. | Treat missing evidence as zero, normal, or comparable. |

## Future Pairwise Gate Outcomes

These are conceptual review outcomes for future pairwise or group-level comparison. They are not current SQL output columns and not current memory slots.

| Outcome | Meaning | Allowed interpretation | Not allowed |
|---|---|---|---|
| `comparable_with_limits` | Two store-period rows share enough context for a narrow comparison. | Compare the specific supported signal while preserving limits. | Transfer a full strategy without checking missing context. |
| `partially_comparable` | Some signals are comparable, but important context differs or is missing. | Explain which signal can be compared and which cannot. | Collapse the result into good or bad. |
| `not_comparable_for_strategy_transfer` | The evidence is too different or incomplete for transferring an operating action. | Refuse or qualify strategy transfer. | Recommend subsidy, price, SKU, or ranking action directly. |

## Hard Rules

1. Do not rank stores as simply best or worst from one metric.
2. Do not infer store stage from one threshold.
3. Do not recommend subsidy changes without promotion cycle, activity context, and competitive-pressure evidence.
4. Do not treat activity_cost_ratio_pct as traditional ROI.
5. Do not treat estimated_income_proxy as audited profit.
6. Do not recompute order_conversion_rate_pct from valid_orders / entry_users.
7. Do not use top-3 SKU evidence as full product-category sales share.
8. Do not compare a missing top-SKU signal as if the store had zero SKU concentration.
9. Do not mix periods casually when activity intensity, ranking, refunds, or competition may differ.
10. Do not let the memory layer produce an operating conclusion without preserving evidence and limitations.

## Role of SQL

The SQL layer should prepare comparable diagnostic evidence.

It should:

- carry through backend-defined fields using canonical names;
- calculate documented SQL-derived ratios and pressure indicators;
- generate comparison-scope and limitation notes;
- avoid inventing unsupported business labels;
- keep source metrics, SQL-derived diagnostics, and memory-facing slots separate.

## Role of the Memory Layer

The memory layer should not decide the business action directly.

It should:

- record each store-period profile;
- preserve source fields and calculation notes;
- remember limitations such as missing promotion-cycle data or incomplete SKU evidence;
- prevent isolated metrics from being reused as general strategy;
- answer with a cautious comparison, qualification, or refusal depending on the available evidence.

## Next Implementation Step

The next implementation step is not to add a more complex AI layer.

The next implementation step is to turn this gate into testable behavior:

1. add a small set of peer-comparison review cases;
2. check whether each answer preserves field definitions;
3. check whether each answer refuses unsupported strategy transfer;
4. expand from B-F to more stores only after the same field contract is preserved.
