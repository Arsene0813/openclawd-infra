# Demo 2 Source Notes

Demo 2 uses anonymized store-period records manually transcribed from the Meituan Waimai merchant backend.

All records use the same reporting window: 2026-03-01 to 2026-03-31.

Included stores:

- Store B
- Store C
- Store D
- Store E
- Store F

Source CSV files:

- demo2_store_period_metrics.csv
- demo2_top_search_terms.csv
- demo2_top_skus_by_sales_volume.csv
- demo2_top_skus_by_transaction_amount.csv

The raw metrics are treated as Meituan backend metrics. SQL-derived diagnostics should not redefine backend fields.

Traffic-source entry metrics are backend-reported source-level metrics. They are not assumed to be mutually exclusive and are not required to sum to entry_users.

activity_cost_ratio_pct follows this project formula:

    activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100

It should not be described as traditional ROI.

Chinese backend values are retained for traceability. English helper columns are added only for readability:

- search_term: original backend search term
- search_term_en: conservative English translation
- sku_name: original backend SKU name
- sku_name_en: conservative English translation

The repository does not include exhaustive Meituan backend screenshots because they contain sensitive store-level operating information.

The project instead uses anonymized structured data, source notes, validation scripts, SQL outputs, generated memory facts, and evaluation results.

Demo 2 currently excludes:

- total SKU count
- full SKU category-share analysis
- SKU-level promotion participation
- SKU original price or activity price
- store rating
- review count
- delivery conditions
- stockout history
- manually counted competitor quantity
