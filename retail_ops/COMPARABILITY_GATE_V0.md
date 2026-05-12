# Comparability Gate V0

This document defines the first review version of the cross-store comparability gate for the retail operations extension.

The gate exists because Meituan merchant-backend data is rich but mainly single-store-oriented. For a multi-store instant-retail business, the harder question is not whether one store improved. The harder question is whether two store-period rows can be compared for a specific operating question before any interpretation or strategy transfer is attempted.

## Current Scope

Current implemented scope:

- Demo 1: Store A month-over-month diagnostic.
- Demo 2: Stores B-F, same March 2026 reporting window, row-level cross-store diagnostic.
- Demo 3: pairwise comparability gate over the current Demo 2 B-F March 2026 output.
- Current system boundary: not a full 48-store decision-support system.

Demo 3 does not classify stores into market-area types. It does not create a final store ranking. It tests whether a pair of store-period rows can be compared for a narrow question.

## Operating Chain

The retail operating chain used by this project is:

```text
being seen -> being entered -> being ordered -> being selected again or maintaining market share
```

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are operating levers inside this chain. They should be read through store context, not treated as isolated goals.

## Gate Question

Before comparing two stores, the system should ask:

```text
Are these two store-period rows comparable enough for the specific question being asked?
```

The answer can differ by question:

- Two stores may be comparable for search-entry structure.
- The same two stores may be only partially comparable for order-quality pressure.
- The same two stores may not be comparable for activity or subsidy strategy transfer.
- A store with missing top-SKU evidence should not be compared on product-mix concentration.
- A region context value should not be treated as a mature market-area classification.

## Region / Market-Context Boundary

`region_type` is not a hard classification field.

In the current project, `region_type` is only weak market-context evidence. It may help readers notice whether two store rows come from the same available region label, but it is not enough to decide market similarity.

The current sample is too small to support a reliable data-driven regional classification. Market-area classification should not be created from subjective experience, intuition, or habitual labels. Future classification should require more stores and stronger evidence such as purchasing power, delivery radius, competition, order structure, price pressure, activity intensity, refund pressure, invalid-order pressure, and SKU evidence.

Therefore, Demo 3 uses `region_type_comparison_note` only as an interpretation note. It must not be used as a hard peer-store grouping rule.

## Gate Dimensions

| Gate dimension | Existing evidence fields | Why it matters | Current boundary |
|---|---|---|---|
| Reporting period alignment | period_start, period_end, period_month, comparison_scope_flag | Different periods may reflect different activity cycles, demand conditions, or competition. | Demo 2 and Demo 3 currently use March 2026 same-period rows. |
| Region and market context | region_type, region_type_comparison_note | Market context may affect demand, competition, delivery radius, purchasing power, and customer behavior. | region_type is weak context only, not a market-area classification. |
| Store operating format | store_type, store_type_comparison_note | Self-operated, warehouse-style, and partner-store contexts may not be directly comparable. | store_type is metadata, not a performance label. |
| Visibility and ranking profile | exposure_users, exposure_times, store_average_rank, search_exposure_users, search_average_rank | A store must first be seen before entry or order signals can be interpreted. | Ranking and exposure do not prove transaction causality by themselves. |
| Entry structure | entry_users, entry_times, search_entry_users, search_entry_rate_pct, search_entry_share_pct, search_entry_share_gap_pct | Search-heavy traffic may behave differently from other traffic sources. | Source-level exposure and entry signals should not be over-summed or over-attributed. |
| Transaction and conversion profile | transaction_amount, transaction_orders, order_users, order_conversion_rate_pct, payment_users, payment_conversion_rate_pct, average_order_value | More orders, higher transaction amount, and stronger conversion are not the same signal. | order_conversion_rate_pct follows the backend definition and must not be recomputed from valid_orders. |
| Activity and subsidy profile | activity_original_transaction_amount, activity_orders, activity_cost, merchant_subsidy_amount, platform_subsidy_amount, activity_cost_ratio_pct, activity_order_share_pct, activity_order_share_gap_pct, activity_cost_ratio_gap_pct | Activity may support exposure, new-store ramp-up, or market-share defense. | Activity data alone does not prove a clean intervention effect. |
| Refund and invalid-order pressure | refund_amount, full_refund_orders, refund_orders_all_or_partial, valid_orders, invalid_orders, refund_pressure_pct, invalid_order_pressure_pct, refund_pressure_gap_pct, invalid_order_pressure_gap_pct | Order growth with refund or cancellation pressure should be interpreted differently from clean order growth. | Refund amount is dated by refund-success date and is not a perfect original-order cohort measure. |
| Top-SKU evidence | sku_name, sku_transaction_amount, sales_volume, top3_sku_transaction_amount_share_pct, top3_sku_concentration_gap_pct | Concentrated leading SKUs may affect transferability of operating signals. | Top-3 SKU evidence is lightweight product-mix evidence, not full product-category sales share. |
| Missing evidence | comparison_limit_notes, pairwise_limit_notes | Missing data should create limits instead of being silently treated as zero. | Missing transaction, order, SKU, activity, or quality evidence should constrain interpretation. |

## Implemented SQL Scope Flags

The current Demo 2 SQL output uses `comparison_scope_flag` as an implemented SQL-derived scope/readiness field.

| `comparison_scope_flag` value | Meaning | Allowed interpretation | Not allowed |
|---|---|---|---|
| `same_period_diagnostic_ready` | The row is inside the current Demo 2 same-period diagnostic scope. | Discuss same-period diagnostic evidence with limits. | Treat as a performance ranking or final operating recommendation. |
| `not_comparable_period_mismatch` | The row does not match the Demo 2 reporting window. | Refuse or qualify same-period comparison. | Mix periods casually as if demand, activity, and competition were aligned. |
| `insufficient_data` | One or more required diagnostic fields are missing. | State the missing evidence and constrain interpretation. | Treat missing evidence as zero, normal, or comparable. |

## Implemented Demo 3 Pairwise Gate Outcomes

The current Demo 3 pairwise output uses `pairwise_comparison_decision` as an implemented SQL-derived pairwise gate field.

| `pairwise_comparison_decision` value | Meaning | Allowed interpretation | Not allowed |
|---|---|---|---|
| `comparable_with_limits` | The pair can be compared for the selected narrow question, with notes preserved. | Compare the supported signal while preserving limitations. | Transfer a full operating strategy. |
| `partially_comparable` | Some evidence can be compared, but important context differs or the gap is too large for a stronger conclusion. | Explain which signal can be compared and which limits remain. | Collapse the pair into good vs bad. |
| `not_comparable_for_strategy_transfer` | The pair is too different or too incomplete for transferring an operating action. | Refuse or qualify strategy transfer. | Recommend subsidy, price, SKU, ranking, or store-stage action directly. |

## Hard Rules

1. Do not rank stores as simply best or worst from one metric.
2. Do not infer store stage from one threshold.
3. Do not recommend subsidy changes without promotion cycle, activity context, and competitive-pressure evidence.
4. Do not treat `activity_cost_ratio_pct` as traditional ROI.
5. Do not treat `estimated_income_proxy` as audited profit.
6. Do not recompute `order_conversion_rate_pct` from `valid_orders / entry_users`.
7. Do not use top-3 SKU evidence as full product-category sales share.
8. Do not compare a missing top-SKU signal as if the store had zero SKU concentration.
9. Do not mix periods casually when activity intensity, ranking, refunds, or competition may differ.
10. Do not treat `region_type` as a mature market-area classification.
11. Do not use `region_type` alone to decide peer-store comparability.
