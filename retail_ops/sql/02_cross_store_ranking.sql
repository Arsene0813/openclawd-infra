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
    RANK() OVER (ORDER BY search_visit_share DESC) AS rank_search_visit_share,

    search_exposure_to_visit_rate,
    RANK() OVER (ORDER BY search_exposure_to_visit_rate DESC) AS rank_search_exposure_to_visit_rate,

    refund_revenue_ratio,
    RANK() OVER (ORDER BY refund_revenue_ratio ASC) AS rank_low_refund_revenue_ratio,

    refund_order_ratio,
    RANK() OVER (ORDER BY refund_order_ratio ASC) AS rank_low_refund_order_ratio,

    subsidy_to_revenue_ratio,
    RANK() OVER (ORDER BY subsidy_to_revenue_ratio ASC) AS rank_low_subsidy_to_revenue_ratio,

    revenue_per_visitor,
    RANK() OVER (ORDER BY revenue_per_visitor DESC) AS rank_revenue_per_visitor

FROM derived
ORDER BY revenue DESC;
