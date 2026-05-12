-- Demo 3: pairwise comparability gate over Demo 2 B-F output
--
-- Input table expected:
--   demo2_cross_store_comparability_output
--
-- This SQL does not rank stores as better or worse.
-- It tests whether a pair of store-period rows can be compared for a narrow question.
-- region_type is used only as weak context and must not be treated as market-area classification.

WITH base AS (
    SELECT
        store_id,
        period_month,
        period_start,
        period_end,
        region_type,
        store_type,
        comparison_scope_flag,
        comparison_limit_notes,
        CAST(search_entry_share_pct AS REAL) AS search_entry_share_pct,
        CAST(activity_order_share_pct AS REAL) AS activity_order_share_pct,
        CAST(activity_cost_ratio_pct AS REAL) AS activity_cost_ratio_pct,
        CAST(refund_pressure_pct AS REAL) AS refund_pressure_pct,
        CAST(invalid_order_pressure_pct AS REAL) AS invalid_order_pressure_pct,
        CAST(top3_sku_transaction_amount_share_pct AS REAL) AS top3_sku_transaction_amount_share_pct
    FROM demo2_cross_store_comparability_output
),
pairs AS (
    SELECT
        a.store_id AS reference_store_id,
        b.store_id AS candidate_store_id,
        a.period_month AS period_month,
        a.period_start AS reference_period_start,
        a.period_end AS reference_period_end,
        b.period_start AS candidate_period_start,
        b.period_end AS candidate_period_end,
        a.region_type AS reference_region_type,
        b.region_type AS candidate_region_type,
        a.store_type AS reference_store_type,
        b.store_type AS candidate_store_type,
        a.comparison_scope_flag AS reference_scope_flag,
        b.comparison_scope_flag AS candidate_scope_flag,
        a.comparison_limit_notes AS reference_limit_notes,
        b.comparison_limit_notes AS candidate_limit_notes,
        ROUND(ABS(a.search_entry_share_pct - b.search_entry_share_pct), 2) AS search_entry_share_gap_pct,
        ROUND(ABS(a.activity_order_share_pct - b.activity_order_share_pct), 2) AS activity_order_share_gap_pct,
        ROUND(ABS(a.activity_cost_ratio_pct - b.activity_cost_ratio_pct), 2) AS activity_cost_ratio_gap_pct,
        ROUND(ABS(a.refund_pressure_pct - b.refund_pressure_pct), 2) AS refund_pressure_gap_pct,
        ROUND(ABS(a.invalid_order_pressure_pct - b.invalid_order_pressure_pct), 2) AS invalid_order_pressure_gap_pct,
        ROUND(ABS(a.top3_sku_transaction_amount_share_pct - b.top3_sku_transaction_amount_share_pct), 2) AS top3_sku_concentration_gap_pct
    FROM base a
    JOIN base b
        ON a.store_id < b.store_id
       AND a.period_month = b.period_month
),
expanded AS (
    SELECT *, 'search_entry_structure' AS comparison_question_type FROM pairs
    UNION ALL
    SELECT *, 'activity_transfer' AS comparison_question_type FROM pairs
    UNION ALL
    SELECT *, 'order_quality_pressure' AS comparison_question_type FROM pairs
),
scored AS (
    SELECT
        reference_store_id,
        candidate_store_id,
        period_month,
        comparison_question_type,
        reference_region_type,
        candidate_region_type,
        reference_store_type,
        candidate_store_type,
        CASE
            WHEN reference_region_type = candidate_region_type
                THEN 'region_type_value_matches_but_not_market_classification'
            ELSE 'region_type_value_differs_and_market_classification_is_unresolved'
        END AS region_type_comparison_note,
        CASE
            WHEN reference_store_type = candidate_store_type
                THEN 'store_type_matches'
            ELSE 'store_type_differs'
        END AS store_type_comparison_note,
        search_entry_share_gap_pct,
        activity_order_share_gap_pct,
        activity_cost_ratio_gap_pct,
        refund_pressure_gap_pct,
        invalid_order_pressure_gap_pct,
        top3_sku_concentration_gap_pct,
        reference_scope_flag,
        candidate_scope_flag,
        reference_limit_notes,
        candidate_limit_notes,
        CASE
            WHEN reference_period_start != candidate_period_start
              OR reference_period_end != candidate_period_end
              OR reference_scope_flag != 'same_period_diagnostic_ready'
              OR candidate_scope_flag != 'same_period_diagnostic_ready'
                THEN 'not_comparable_for_strategy_transfer'

            WHEN comparison_question_type = 'search_entry_structure'
             AND search_entry_share_gap_pct <= 15
                THEN 'comparable_with_limits'

            WHEN comparison_question_type = 'search_entry_structure'
                THEN 'partially_comparable'

            WHEN comparison_question_type = 'activity_transfer'
             AND (
                    activity_order_share_gap_pct > 15
                 OR activity_cost_ratio_gap_pct > 10
                 OR refund_pressure_gap_pct > 5
                 OR invalid_order_pressure_gap_pct > 4
                 OR reference_store_type != candidate_store_type
                )
                THEN 'not_comparable_for_strategy_transfer'

            WHEN comparison_question_type = 'activity_transfer'
                THEN 'comparable_with_limits'

            WHEN comparison_question_type = 'order_quality_pressure'
             AND refund_pressure_gap_pct <= 5
             AND invalid_order_pressure_gap_pct <= 4
                THEN 'comparable_with_limits'

            WHEN comparison_question_type = 'order_quality_pressure'
                THEN 'partially_comparable'

            ELSE 'partially_comparable'
        END AS pairwise_comparison_decision
    FROM expanded
)
SELECT
    reference_store_id,
    candidate_store_id,
    period_month,
    comparison_question_type,
    reference_region_type,
    candidate_region_type,
    region_type_comparison_note,
    reference_store_type,
    candidate_store_type,
    store_type_comparison_note,
    search_entry_share_gap_pct,
    activity_order_share_gap_pct,
    activity_cost_ratio_gap_pct,
    refund_pressure_gap_pct,
    invalid_order_pressure_gap_pct,
    top3_sku_concentration_gap_pct,
    pairwise_comparison_decision,
    TRIM(
        region_type_comparison_note || '; ' ||
        store_type_comparison_note || '; ' ||
        CASE
            WHEN comparison_question_type = 'search_entry_structure'
                THEN 'compare_search_entry_structure_only; '
            WHEN comparison_question_type = 'activity_transfer'
                THEN 'activity_strategy_transfer_requires_activity_cost_refund_invalid_order_and_competition_context; '
            WHEN comparison_question_type = 'order_quality_pressure'
                THEN 'compare_refund_and_invalid_order_pressure_only; '
            ELSE ''
        END ||
        CASE
            WHEN search_entry_share_gap_pct > 15
                THEN 'large_search_entry_share_gap; '
            ELSE ''
        END ||
        CASE
            WHEN activity_order_share_gap_pct > 15
                THEN 'large_activity_order_share_gap; '
            ELSE ''
        END ||
        CASE
            WHEN activity_cost_ratio_gap_pct > 10
                THEN 'large_activity_cost_ratio_gap; '
            ELSE ''
        END ||
        CASE
            WHEN refund_pressure_gap_pct > 5
                THEN 'large_refund_pressure_gap; '
            ELSE ''
        END ||
        CASE
            WHEN invalid_order_pressure_gap_pct > 4
                THEN 'large_invalid_order_pressure_gap; '
            ELSE ''
        END ||
        CASE
            WHEN top3_sku_concentration_gap_pct > 10
                THEN 'large_top3_sku_concentration_gap_but_not_full_category_share; '
            ELSE ''
        END ||
        'do_not_rank_stores_or_transfer_strategy_directly'
    ) AS pairwise_limit_notes
FROM scored
ORDER BY reference_store_id, candidate_store_id, comparison_question_type;
