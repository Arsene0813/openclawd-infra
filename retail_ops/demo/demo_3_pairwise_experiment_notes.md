# Demo 3: Pairwise Experiment Notes

This note gives concrete examples from the Demo 3 pairwise comparability gate.

The goal is not to prove that one store is better than another. The goal is to show how the gate decides whether two store-period rows can be compared for a narrow operating question.

Source output:

- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`

Current supported `comparison_question_type` values:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

## Why this note exists

The Meituan backend gives many metrics, but cross-store comparison can easily become misleading.

Two stores can look similar on one metric and still differ too much in activity involvement, refund pressure, invalid-order pressure, store type, or product-mix concentration. Demo 3 makes that problem explicit by testing one store pair and one question type at a time.

## Example 1: Store B vs Store D for `search_entry_structure`

| Field | Demo 3 value |
|---|---:|
| `reference_store_id` | B |
| `candidate_store_id` | D |
| `comparison_question_type` | `search_entry_structure` |
| `search_entry_share_gap_pct` | 1.05 |
| `activity_order_share_gap_pct` | 5.21 |
| `activity_cost_ratio_gap_pct` | 9.20 |
| `refund_pressure_gap_pct` | 1.81 |
| `invalid_order_pressure_gap_pct` | 0.45 |
| `top3_sku_concentration_gap_pct` | 5.75 |
| `pairwise_comparison_decision` | `comparable_with_limits` |

Interpretation:

B and D are usable for a narrow search-entry-structure comparison because the search-entry share gap is small. This means the two rows can be discussed for how much store entry depends on search traffic.

The comparison is still limited. `region_type` differs and the project does not treat `region_type` as a mature market-area classification. The answer should therefore discuss search-entry structure only, not claim that the two stores share the same market environment.

What this prevents:

It prevents a weak conclusion such as "B and D are comparable stores overall." The gate only supports a narrower statement: B and D can be compared for search-entry structure, with region-context limits preserved.

## Example 2: Store B vs Store E for `activity_transfer`

| Field | Demo 3 value |
|---|---:|
| `reference_store_id` | B |
| `candidate_store_id` | E |
| `comparison_question_type` | `activity_transfer` |
| `search_entry_share_gap_pct` | 1.69 |
| `activity_order_share_gap_pct` | 19.64 |
| `activity_cost_ratio_gap_pct` | 5.26 |
| `refund_pressure_gap_pct` | 8.34 |
| `invalid_order_pressure_gap_pct` | 6.75 |
| `top3_sku_concentration_gap_pct` | 1.40 |
| `pairwise_comparison_decision` | `not_comparable_for_strategy_transfer` |

Interpretation:

B and E may look close on search-entry structure, but they are not safe for activity-strategy transfer under the current evidence contract.

The activity-order-share gap is large, refund-pressure gap is large, invalid-order-pressure gap is large, and store type differs. That means activity evidence cannot be copied from one store to the other without additional context.

What this prevents:

It prevents a tempting but unsafe conclusion such as "Store E should copy Store B's activity strategy." The current evidence only supports saying that the pair is not comparable for activity transfer.

## Example 3: Store E vs Store F for `order_quality_pressure`

| Field | Demo 3 value |
|---|---:|
| `reference_store_id` | E |
| `candidate_store_id` | F |
| `comparison_question_type` | `order_quality_pressure` |
| `search_entry_share_gap_pct` | 1.90 |
| `activity_order_share_gap_pct` | 12.59 |
| `activity_cost_ratio_gap_pct` | 17.22 |
| `refund_pressure_gap_pct` | 8.48 |
| `invalid_order_pressure_gap_pct` | 3.75 |
| `top3_sku_concentration_gap_pct` | 6.78 |
| `pairwise_comparison_decision` | `partially_comparable` |

Interpretation:

E and F can be discussed for order-quality pressure only with limits. The two stores have the same `store_type`, and the invalid-order-pressure gap is not as large as the refund-pressure gap. However, the refund-pressure gap is large, and the activity-cost-ratio gap is also large.

The answer should therefore focus on refund and invalid-order pressure instead of turning the comparison into a full activity or store-performance recommendation.

What this prevents:

It prevents the answer layer from hiding the refund-pressure difference behind a general "same store type" statement. It also prevents the answer from treating order-quality comparison as proof of customer satisfaction or fulfillment quality.

## Field-boundary reminder

These examples use Demo 3 SQL-derived pairwise fields. They do not rename the original Meituan backend fields.

Important boundaries:

- `search_entry_share_gap_pct` is based on `search_entry_share_pct`.
- `activity_order_share_gap_pct` is based on `activity_order_share_pct`.
- `activity_cost_ratio_gap_pct` is based on `activity_cost_ratio_pct`.
- `refund_pressure_gap_pct` is based on `refund_pressure_pct`.
- `invalid_order_pressure_gap_pct` is based on `invalid_order_pressure_pct`.
- `top3_sku_concentration_gap_pct` is based on `top3_sku_transaction_amount_share_pct`.

The gate output is a comparison boundary, not a final recommendation.
