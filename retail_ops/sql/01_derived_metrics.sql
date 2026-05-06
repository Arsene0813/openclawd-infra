SELECT
    store_id,
    period,
    region_type,
    store_type,
    business_district_rank,
    revenue,
    valid_orders,
    avg_order_value,
    store_visitors,
    order_conversion_rate,
    search_exposure,
    search_visitors,

    ROUND(1.0 * search_visitors / NULLIF(store_visitors, 0), 4)
        AS search_visit_share,

    ROUND(1.0 * search_visitors / NULLIF(search_exposure, 0), 4)
        AS search_exposure_to_visit_rate,

    refund_amount,
    refund_orders,

    ROUND(1.0 * refund_amount / NULLIF(revenue, 0), 4)
        AS refund_revenue_ratio,

    ROUND(1.0 * refund_orders / NULLIF(valid_orders, 0), 4)
        AS refund_order_ratio,

    promotion_gmv,
    merchant_subsidy,

    ROUND(1.0 * promotion_gmv / NULLIF(revenue, 0), 4)
        AS promotion_gmv_to_revenue_ratio,

    ROUND(1.0 * merchant_subsidy / NULLIF(revenue, 0), 4)
        AS subsidy_to_revenue_ratio,

    ROUND(1.0 * merchant_subsidy / NULLIF(promotion_gmv, 0), 4)
        AS subsidy_to_promotion_gmv_ratio,

    ROUND(1.0 * revenue / NULLIF(store_visitors, 0), 2)
        AS revenue_per_visitor,

    top10_products
FROM store_monthly_metrics
ORDER BY revenue DESC;
