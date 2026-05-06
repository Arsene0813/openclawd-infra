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

    ROUND(1.0 * search_visitors / store_visitors, 4) AS search_visit_share,
    ROUND(1.0 * search_visitors / search_exposure, 4) AS search_exposure_to_visit_rate,

    refund_amount,
    refund_orders,
    ROUND(1.0 * refund_amount / revenue, 4) AS refund_revenue_ratio,
    ROUND(1.0 * refund_orders / valid_orders, 4) AS refund_order_ratio,

    promotion_gmv,
    merchant_subsidy,
    ROUND(1.0 * promotion_gmv / revenue, 4) AS promotion_gmv_to_revenue_ratio,
    ROUND(1.0 * merchant_subsidy / revenue, 4) AS subsidy_to_revenue_ratio,
    ROUND(1.0 * merchant_subsidy / promotion_gmv, 4) AS subsidy_to_promotion_gmv_ratio,

    ROUND(1.0 * revenue / store_visitors, 2) AS revenue_per_visitor,

    top10_products

FROM store_monthly_metrics;
