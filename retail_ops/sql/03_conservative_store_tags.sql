WITH derived AS (
    SELECT
        store_id,
        period,
        region_type,
        store_type,
        business_district_rank,

        revenue,
        valid_orders,
        avg_order_value,

        ROUND(1.0 * search_visitors / store_visitors, 4) AS search_visit_share,
        ROUND(1.0 * search_visitors / search_exposure, 4) AS search_exposure_to_visit_rate,
        ROUND(1.0 * refund_amount / revenue, 4) AS refund_revenue_ratio,
        ROUND(1.0 * refund_orders / valid_orders, 4) AS refund_order_ratio,
        ROUND(1.0 * promotion_gmv / revenue, 4) AS promotion_gmv_to_revenue_ratio,
        ROUND(1.0 * merchant_subsidy / revenue, 4) AS subsidy_to_revenue_ratio,
        ROUND(1.0 * merchant_subsidy / promotion_gmv, 4) AS subsidy_to_promotion_gmv_ratio,
        ROUND(1.0 * revenue / store_visitors, 2) AS revenue_per_visitor

    FROM store_monthly_metrics
)

SELECT
    store_id,
    region_type,
    store_type,
    business_district_rank,
    revenue,
    valid_orders,

    search_visit_share,
    search_exposure_to_visit_rate,
    refund_revenue_ratio,
    refund_order_ratio,
    subsidy_to_revenue_ratio,
    revenue_per_visitor,

    CASE
        WHEN search_visit_share >= 0.85 THEN 'search-heavy'
        WHEN search_visit_share >= 0.70 THEN 'moderately-search-driven'
        ELSE 'less-search-dependent'
    END AS traffic_pattern_tag,

    CASE
        WHEN search_exposure_to_visit_rate >= 0.20 THEN 'high-search-exposure-efficiency'
        WHEN search_exposure_to_visit_rate >= 0.12 THEN 'medium-search-exposure-efficiency'
        ELSE 'low-search-exposure-efficiency'
    END AS search_efficiency_tag,

    CASE
        WHEN refund_revenue_ratio >= 0.15 THEN 'refund-pressure-high'
        WHEN refund_revenue_ratio >= 0.10 THEN 'refund-pressure-medium'
        ELSE 'refund-pressure-low'
    END AS refund_pressure_tag,

    CASE
        WHEN refund_order_ratio >= 0.12 THEN 'refund-order-ratio-high'
        WHEN refund_order_ratio >= 0.08 THEN 'refund-order-ratio-medium'
        ELSE 'refund-order-ratio-low'
    END AS refund_order_tag,

    CASE
        WHEN subsidy_to_revenue_ratio >= 0.40 THEN 'subsidy-intensity-high'
        WHEN subsidy_to_revenue_ratio >= 0.25 THEN 'subsidy-intensity-medium'
        ELSE 'subsidy-intensity-low'
    END AS subsidy_intensity_tag,

    CASE
        WHEN revenue_per_visitor >= 20 THEN 'high-visitor-value'
        WHEN revenue_per_visitor >= 14 THEN 'medium-visitor-value'
        ELSE 'low-visitor-value'
    END AS visitor_value_tag

FROM derived
ORDER BY revenue DESC;
