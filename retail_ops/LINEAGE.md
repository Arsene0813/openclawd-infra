# Retail Data Lineage

The field names in this file follow the implemented retail data contract used across the source CSV, SQL diagnostic, SQL output, and metric dictionary:

- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/outputs/store_a_demo1_sql_output.csv`
- `retail_ops/data/DATA_DICTIONARY.md`

Existing Meituan backend metrics are kept under a single canonical English field name. This avoids mixing multiple English names for the same Chinese backend metric.

## Demo 1 Scope

| Item | Value |
|---|---|
| Store | Store A |
| Period | February 2026 to April 2026 |
| ISO Date Range | 2026-02-01 to 2026-04-30 |
| Data Source | Manually exported Meituan merchant-backend metrics |
| Processing Method | Offline SQL diagnostic |
| Output | SQL-derived CSV, markdown diagnostic, generated retail memory facts |
| Limitation | Single-store demo; not a causal experiment; not cross-store comparison |

Demo 1 interprets April 2026 as the latest observed month in the current Store A sample. SQL boolean diagnostics are evaluated against the latest `period_start` per `store_id`, not against a hard-coded calendar month.

Month-window convention for Store A Demo 1: `2026-02` = `2026-02-01` to `2026-02-28`; `2026-03` = `2026-03-01` to `2026-03-31`; `2026-04` = `2026-04-01` to `2026-04-30`.

## Operating Logic

This demo does not use SQL to assign a fixed operating label to Store A.

In this project, Meituan instant-retail stores are understood through a chain of operating conditions:

```text
being seen -> being entered -> being ordered -> being selected again or maintaining market share
```

Promotion, subsidy, price adjustment, SKU mix, ranking position, and fulfillment quality are treated as operating levers inside this chain.

Short-term ROI is not always the primary target. A new store may use activity subsidy to gain exposure and first orders. A store under external price pressure may use price or activity tools to defend visibility and market share. A store with enough traffic but weak conversion requires a different interpretation from a store with order growth but refund pressure.

Therefore, SQL should prepare comparison-ready diagnostic signals. It should not turn one metric threshold into a fixed store-stage label.

## Claim-to-Data Lineage

| Claim / Diagnostic | Source Fields | SQL Output / Derived Metric | Memory Slot | Limitation |
|---|---|---|---|---|
| Store A's visibility and entry structure can be described from exposure, ranking, entry, and search-entry metrics. | `exposure_users`, `store_average_rank`, `entry_users`, `search_exposure_users`, `search_average_rank`, `search_entry_users` | `search_exposure_share_pct`, `search_entry_share_pct`, `search_entry_rate_pct` | `visibility_entry_profile` | Describes whether the store was being seen and entered; does not prove causal growth. |
| Store A's activity metrics should be interpreted as operating-lever evidence. | `activity_original_transaction_amount`, `activity_orders`, `activity_cost`, `merchant_subsidy_amount`, `platform_subsidy_amount` | `activity_order_share_pct`, `activity_cost_ratio_pct`, `merchant_subsidy_share_of_activity_cost_pct` | `activity_lever_profile` | Activity is a tool inside the operating chain, not a standalone causal explanation or simple ROI judgment. |
| Store A's transaction and conversion signals moved in different directions. | `transaction_amount`, `transaction_orders`, `order_conversion_rate_pct`, `average_order_value` | `transaction_amount_mom_pct`, `transaction_orders_mom_pct`, `average_order_value_mom_pct`, `transaction_recovered_with_conversion_aov_tradeoff` | `transaction_conversion_profile` | Transaction recovery can coexist with weaker conversion or lower AOV. |
| Store A's refund and invalid-order pressure improved in April. | `refund_amount`, `transaction_amount`, `valid_orders`, `invalid_orders` | `refund_pressure_pct`, `invalid_order_pressure_pct`, `refund_pressure_improved` | `order_quality_pressure_profile` | Refund amount is dated by refund success date, not original order cohort. |
| Store A's changes should not be explained by one metric alone. | Visibility, entry, transaction, conversion, activity, refund, invalid-order, and SKU evidence | Combined multi-signal interpretation | `single_metric_attribution_guard` | The demo supports structured comparison of signals, not causal attribution. |
| Top SKU mix appears care-solution-heavy. | Top-3 SKU records | Top-3 SKU observation | `top3_sku_product_mix_note` | Top-3 evidence only; not full SKU category-share analysis. |

## Interpretation Rule

The retail memory layer should prefer cautious answers:

- It may describe observed patterns.
- It may identify operating conditions and data limitations.
- It may warn against unsupported attribution.
- It should refuse or qualify cross-store, causal, margin, store-stage, or SKU-category claims not supported by the current demo data.

## Bilingual Metric Consistency and Lineage Rules

The retail demo uses backend-reported funnel and operation metrics directly when exact numerator, denominator, deduplication rule, attribution rule, or order-status scope may differ.

当分子、分母、去重规则、归因规则或订单状态口径可能不一致时，本 retail demo 直接使用后台展示的漏斗指标和经营指标。

### 1. Conversion-Rate Lineage / 下单转化率口径

English:

`order_conversion_rate_pct` follows the backend business definition:

```text
order_conversion_rate_pct = order_users / entry_users * 100
```

It is not derived from:

```text
valid_orders / entry_users
```

because `valid_orders` is an order-status metric, while `order_users` / 下单人数 is a user-level funnel metric.

中文：

`order_conversion_rate_pct` / 下单转化率遵循后台业务定义：

```text
下单转化率 = 下单人数 / 入店人数
```

它不是由以下公式推导：

```text
有效订单数 / 入店人数
```

原因是：`valid_orders` / 有效订单数是订单状态口径，而 `order_users` / 下单人数是用户级漏斗口径。

### 2. Traffic-Source Lineage / 流量来源口径

English:

Traffic-source users may overlap. The same customer may see the store through multiple exposure sources, so source-level exposure users should not be summed into total exposure users.

`search_entry_users / entry_users` is used only as a directional source-entry structure signal.

中文：

不同流量来源用户可能重叠。同一个顾客可能通过多个曝光来源看到门店，因此来源级曝光人数不能直接相加成总曝光人数。

`search_entry_users / entry_users` 只作为方向性的来源入店结构信号。

### 3. Activity / Promotion Lineage / 活动与促销口径

English:

`activity_cost_ratio_pct` follows the backend formula:

```text
activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100
```

A smaller value means lower activity cost per unit of activity-driven revenue. The project avoids calling this traditional ROI because traditional ROI is often interpreted in the opposite direction.

Activity and subsidy should be interpreted as operating levers. Their meaning depends on store stage, competitive pressure, ranking pressure, product mix, and order-quality signals.

中文：

`activity_cost_ratio_pct` 遵循后台公式：

```text
投入产出比 = 活动成本 / 活动营业总额
```

因为这是成本除以活动带动营业额，所以数值越小，单位活动营业额对应的成本越低。

本项目避免将其称为传统 ROI，因为传统 ROI 通常方向相反，往往是收益除以成本且数值越大越好。

活动与补贴应解释为经营工具。它们的意义取决于门店阶段、竞争压力、排名压力、商品结构和订单质量信号。

### 4. Transaction Metric Lineage / 成交指标口径

English:

`transaction_amount` and `transaction_orders` refer to same-day paid and same-day not-cancelled orders.

For the transaction metric page:

```text
average_order_value = transaction_amount / transaction_orders
```

If another backend page defines 单均价 using valid orders, it should be treated as a separate backend-reported metric rather than mixed with transaction fields.

中文：

`transaction_amount` / 成交金额 与 `transaction_orders` / 成交订单量 指当天支付且当天未取消的订单口径。

在成交指标页面中：

```text
单均价 = 成交金额 / 成交订单量
```

如果另一个后台页面使用有效订单定义单均价，应将其作为另一个后台展示指标处理，不应与成交字段混用。

### 5. Estimated Income Lineage / 预计收入口径

English:

`estimated_income_proxy` is treated as a platform-displayed income proxy. It should not be interpreted as audited profit because the current demo does not contain the full platform calculation breakdown.

中文：

`estimated_income_proxy` 被视为平台展示的预计收入 proxy。由于当前 demo 不包含平台完整计算拆解，因此不应解释为审计利润或真实净利润。

### 6. Refund Lineage / 退款口径

English:

`refund_amount` is counted by refund-success date. It is interpreted as refund pressure during the selected period, not as a perfect refund rate for the original order cohort.

中文：

`refund_amount` / 退款金额 按退款成功日期统计。它应被解释为所选周期内的退款压力，而不是原始订单 cohort 的精确退款率。

### 7. Ranking Lineage / 排名口径

English:

Business-district ranking is only comparable among merchants in the same main category and business district. Ranking may be unavailable when the store has no honeycomb/grid information or no sales activity.

中文：

商圈排名只在同一商圈、同一主营品类商家之间可比。当门店无蜂窝信息或无动销时，商圈排名可能不可用。

## SKU Evidence Grain Note

Top-SKU evidence uses SKU-level fields from `retail_ops/data/store_a_top_skus.csv`.

- `sku_transaction_amount` is SKU-period-level transaction evidence.
- It must not be confused with store-period-level `transaction_amount`.
- Top-SKU evidence is used only as lightweight product-mix support, not as full category-level sales-share analysis.

## Demo 2 Cross-Store Comparability Lineage

Demo 2 extends the retail operations prototype from a single-store month-over-month diagnostic to a same-period cross-store diagnostic.

The current Demo 2 scope is limited to five anonymized stores: B, C, D, E, and F.

All Demo 2 records use the same reporting window:

- period_start: 2026-03-01
- period_end: 2026-03-31
- period_month: 2026-03

Demo 2 does not rank stores as simply better or worse. Its purpose is to structure comparable backend metrics, derive cautious diagnostic signals, and preserve interpretation limits before any operating recommendation is made.

### Source data

Demo 2 source data is stored in:

- retail_ops/data/demo2_store_period_metrics.csv
- retail_ops/data/demo2_top_search_terms.csv
- retail_ops/data/demo2_top_skus_by_sales_volume.csv
- retail_ops/data/demo2_top_skus_by_transaction_amount.csv
- retail_ops/data/demo2_source_notes.md

The source metrics are manually transcribed from the Meituan Waimai merchant backend and anonymized at the store level.

Original Chinese backend search terms and SKU names are retained for traceability. English helper columns are included only for readability.

### SQL diagnostic output

Demo 2 SQL is stored in:

- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

The generated SQL output is stored in:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

The Demo 2 output intentionally separates carried-through canonical fields from SQL-derived diagnostic fields.

Carried-through canonical or backend-formula fields include:

- `region_type`
- `store_type`
- `business_district_rank`
- `activity_cost_ratio_pct`

`activity_cost_ratio_pct` follows the documented backend-formula interpretation:

- `activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100`

It is kept in the SQL output because it is important operating-lever evidence, but it should not be described as a newly invented Demo 2 diagnostic or as traditional ROI.

SQL-derived diagnostic fields include:

- `search_entry_rate_pct = search_entry_users / search_exposure_users * 100`
- `search_entry_share_pct = search_entry_users / entry_users * 100`
- `activity_order_share_pct = activity_orders / transaction_orders * 100`
- `refund_pressure_pct = refund_amount / transaction_amount * 100`
- `invalid_order_pressure_pct = invalid_orders / (valid_orders + invalid_orders) * 100`
- `top3_sku_transaction_amount_share_pct = top3_sku_transaction_amount / transaction_amount * 100`
- `comparison_scope_flag`
- `comparison_limit_notes`

These derived fields are diagnostic summaries. They do not replace Meituan backend definitions, rank stores, assign store stages, or prove causal operating effects.

### Claim-to-field mapping

Claim: stores are in the same Demo 2 reporting window.

Supporting fields:

- period_month
- period_start
- period_end

Interpretation limit:

- Same period alignment improves comparability but does not remove differences in region, store type, activity conditions, competition, fulfillment, or SKU mix.

Claim: visibility and entry can be compared cautiously across stores.

Supporting fields:

- exposure_users
- exposure_times
- store_average_rank
- entry_users
- entry_times
- entry_conversion_rate_pct
- search_exposure_users
- search_average_rank
- search_entry_users
- search_entry_rate_pct
- search_entry_share_pct

Interpretation limit:

- Traffic-source entry metrics are backend-reported source-level metrics and are not assumed to be mutually exclusive.
- Visibility and entry metrics do not prove causal transaction growth.

Claim: activity involvement should constrain cross-store transaction comparison.

Supporting fields:

- activity_original_transaction_amount
- activity_orders
- activity_cost
- merchant_subsidy_amount
- platform_subsidy_amount
- activity_cost_ratio_pct
- activity_order_share_pct

Interpretation limit:

- Activity mechanism details and promotion cycle dates are not included.
- activity_cost_ratio_pct is a cost ratio, not traditional ROI.
- Activity metrics describe tool usage, not causal proof.

Claim: refund and invalid-order pressure should constrain direct performance comparison.

Supporting fields:

- refund_amount
- full_refund_orders
- refund_orders_all_or_partial
- valid_orders
- invalid_orders
- transaction_orders
- refund_pressure_pct
- invalid_order_pressure_pct

Interpretation limit:

- refund_amount is counted by refund-success date.
- refund_pressure_pct is not an exact original-order cohort refund rate.
- invalid-order reasons are not included.

Claim: top search terms provide lightweight demand evidence.

Supporting fields:

- search_term
- search_term_en
- search_term_exposure_times
- search_term_click_times
- search_term_order_times

Interpretation limit:

- Top search terms are store-period evidence, not complete regional consumer-preference proof.
- English search terms are helper translations, not backend source values.

Claim: top SKU evidence provides lightweight product-mix evidence.

Supporting fields:

- sku_name
- sku_name_en
- sku_transaction_amount
- sales_volume
- sku_category_note
- top3_sku_transaction_amount
- top3_sku_transaction_amount_share_pct

Interpretation limit:

- Top-3 SKU evidence is not full SKU category-share analysis.
- Demo 2 does not perform full manual SKU category classification.
- English SKU names are helper translations, not backend source values.

### Memory fact output

Demo 2 generated memory facts are stored in:

- retail_ops/outputs/generated_demo2_retail_memory_facts.json

The generation script is:

- retail_ops/scripts/generate_demo2_retail_memory_facts.py

The validation script is:

- retail_ops/scripts/validate_demo2_retail_memory_facts.py

Demo 2 reuses existing canonical retail memory slots:

- visibility_entry_profile
- activity_lever_profile
- transaction_conversion_profile
- order_quality_pressure_profile
- top3_sku_product_mix_note
- single_metric_attribution_guard

Demo 2 does not introduce store-stage labels or best-store rankings.

## Demo 2 Carry-Through Note: Order and Payment Amount Fields

This patch carries `order_amount` and `payment_amount` from `retail_ops/data/demo2_store_period_metrics.csv` into `retail_ops/outputs/demo2_cross_store_comparability_output.csv` and `retail_ops/outputs/generated_demo2_retail_memory_facts.json`.

Lineage:

- `demo2_store_period_metrics.csv`
- `02_demo2_cross_store_comparability.sql`
- `demo2_cross_store_comparability_output.csv`
- `generate_demo2_retail_memory_facts.py`
- `generated_demo2_retail_memory_facts.json`

Interpretation boundary:

- `order_amount` is read with `order_users`, `order_times`, and `order_conversion_rate_pct`.
- `payment_amount` is read with `payment_users` and `payment_conversion_rate_pct`.
- `transaction_amount` remains a separate transaction metric and should not be merged with order-submission or payment-funnel amount fields.

---

## Future Pairwise Comparability Gate Pairwise Comparability Gate Lineage

A future pairwise comparability gate would extend Demo 2 from same-period cross-store diagnostics into pairwise comparability checks.

The input is:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

The output is:

A future pairwise comparability gate should not compare all stores globally. It should check whether two store-period rows can be compared for one narrow question at a time.

### Supported question types

Current supported `comparison_question_type` values are:

- `order_quality_pressure`

These values must stay consistent across:

- `retail_ops/data/DATA_DICTIONARY.md`

### Claim-to-field checks

Claim: two stores can be checked as a pair for one narrow question.

Supporting fields:

- `reference_store_id`
- `candidate_store_id`
- `comparison_question_type`

Interpretation limit:

- This is not a full peer-grouping model.
- This is not a global store ranking.

Claim: the two stores may share or differ in weak region context.

Supporting fields:

- `reference_region_type`
- `candidate_region_type`
- `region_type_comparison_note`

Interpretation limit:

- `region_type` is weak context only.
- It is not a market-area classification.
- It is not a hard comparability condition.

Claim: the two stores may share or differ in operating format.

Supporting fields:

- `reference_store_type`
- `candidate_store_type`
- `store_type_comparison_note`

Interpretation limit:

- `store_type` can constrain comparison.
- It does not decide performance quality by itself.

Claim: search-entry structure is close or far apart.

Supporting field:

- `search_entry_share_gap_pct`

Interpretation limit:

- This supports only a narrow traffic-structure comparison.
- It does not prove that search caused order growth.

Claim: activity involvement differs between two stores.

Supporting field:

- `activity_order_share_gap_pct`

Interpretation limit:

- Activity order share is operating-lever evidence.
- It does not prove that promotion caused performance differences.

Claim: activity cost structure differs between two stores.

Supporting field:

- `activity_cost_ratio_gap_pct`

Interpretation limit:

- This is activity-cost-ratio evidence.
- It is not traditional ROI.
- It is not profit margin.

Claim: refund pressure differs between two stores.

Supporting field:

- `refund_pressure_gap_pct`

Interpretation limit:

- Refund amount is counted by refund-success date.
- It is not a perfect original-order cohort refund rate.

Claim: invalid-order pressure differs between two stores.

Supporting field:

- `invalid_order_pressure_gap_pct`

Interpretation limit:

- Cancelled-order pressure may reflect fulfillment, stock, customer, or operational differences not visible in the current sample.

Claim: top-SKU concentration differs between two stores.

Supporting field:

- `top3_sku_concentration_gap_pct`

Interpretation limit:

- This is top-three SKU concentration evidence.
- It is not full product-category sales share.

Claim: a pair can be compared, partially compared, or refused for the selected question.

Supporting fields:

Interpretation limit:

- This is a comparability gate.
- It is not a final operating recommendation.

### Directionality boundary

Future pairwise comparability rows should be read as comparability checks, not directional strategy-transfer decisions.

A later answer layer should avoid claims such as:

- copy Store B's promotion strategy to Store E;
- Store B is better than Store E;
- Stores B and E belong to the same market type;
- the activity strategy caused the performance difference.

Supported wording is narrower:

- whether the pair is comparable for the selected question;
- which gap fields support or limit the comparison;
- which missing or unresolved context prevents stronger advice.

### Future Pairwise Comparability Gate answer boundary

- `reference_store_id`
- `candidate_store_id`
- `comparison_question_type`
- relevant gap fields

If a user asks for a full 48-store ranking, market-area classification, causal promotion effect, or direct strategy recommendation, the answer should qualify or refuse the request unless future evidence has been added and documented.

## Future Comparability-Gate Lineage

The comparability gate is not currently implemented as a finished demo.

Current implemented retail lineage stops at Demo 2:

- selected Meituan backend fields;
- DATA_DICTIONARY.md definitions;
- canonical CSV files;
- Demo 1 and Demo 2 SQL diagnostics;
- Demo 1 and Demo 2 output CSV files;
- validation and evaluation for the implemented scope.

Future pairwise comparability lineage should only be added after broader multi-store evidence is available. It should document how transaction order volume, transaction amount, activity or promotion status, store type, region and market context, competition environment, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows are used.

At the current sample size, region_type remains weak context only and must not be used as a hard market-area classification or peer-store grouping rule.
