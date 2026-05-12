# Project Status

This file records the current implementation boundary of the repository.

Admissions-facing narrative belongs in `PROJECT_SUMMARY_FOR_ADMISSIONS.md`.

The technical overview belongs in `README.md`.

Detailed retail evidence belongs in the retail demo files, dictionary, lineage, SQL outputs, generated facts, and evaluation files.

## Current Implemented Scope

| Area | Status | Evidence |
|---|---:|---|
| Livestream memory layer | Implemented prototype | `api/main.py`, `eval/eval_report.md`, `eval/results/eval_result_11_pass.txt` |
| Typed memory and lifecycle-aware retrieval | Implemented prototype | fact policy registry, retrieval gating, stale / unsupported fallback behavior |
| Retail Demo 1 | Implemented Store A month-over-month diagnostic | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` |
| Retail Demo 2 | Implemented limited B-F same-period cross-store comparability diagnostic | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` |
| Retail Demo 3 | Implemented limited B-F pairwise comparability gate | `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv` |
| Retail source metric dictionary | Implemented | `retail_ops/data/DATA_DICTIONARY.md` |
| Retail claim-to-field lineage | Implemented for current retail demos | `retail_ops/LINEAGE.md` |
| Store A SQL diagnostic output | Implemented | `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`, `retail_ops/outputs/store_a_demo1_sql_output.csv` |
| Demo 2 SQL diagnostic output | Implemented | `retail_ops/sql/02_demo2_cross_store_comparability.sql`, `retail_ops/outputs/demo2_cross_store_comparability_output.csv` |
| Demo 3 SQL pairwise output | Implemented | `retail_ops/sql/03_demo2_pairwise_comparability_gate.sql`, `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv` |
| Retail memory-facing facts | Implemented for Store A Demo 1 and Demo 2 B-F facts | `retail_ops/outputs/generated_retail_memory_facts.json`, `retail_ops/outputs/generated_demo2_retail_memory_facts.json` |
| Retail validation | Implemented | retail data-contract validation, Demo 2 validation scripts, Demo 3 pairwise output validation, project consistency validation |
| Retail retrieval / offline evaluation | Implemented for current supported scopes | `eval/eval_retail.py`, Store A retrieval eval, Demo 2 facts eval, Demo 2 comparability eval, Demo 2 answer-boundary eval, Demo 3 pairwise gate eval |
| Retail API endpoints | Implemented for Demo 1 and Demo 2 only | `/chat_retail_ops_kb`, `/chat_retail_ops_demo2_kb` |

## Current Boundary

The retail extension now has three implemented stages.

Demo 1 demonstrates how Meituan backend metrics can be normalized, checked, traced, and converted into memory-facing facts for a Store A month-over-month diagnostic.

Demo 2 adds a limited same-period cross-store diagnostic for anonymized Stores B-F using March 2026 backend data. It structures cross-store source metrics, derives comparability diagnostics, generates Demo 2 retail memory facts, validates those facts, and exposes a separate file-backed Demo 2 API endpoint.

Demo 3 adds a pairwise comparability gate over the current Demo 2 B-F March 2026 output. It tests whether store pairs can be compared for narrow questions:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

Demo 3 is currently implemented as SQL output, saved CSV output, documentation, validation, and offline evaluation. It is not yet exposed through a retrieval endpoint.

The current implementation does not yet perform full multi-store comparability grouping across all 48 stores.

## Not Yet Implemented

| Area | Current boundary |
|---|---|
| Automated Meituan backend ingestion | Not implemented |
| Full 48-store cross-store comparison | Not implemented |
| Store-stage classification across all stores | Not implemented; Demo 3 deliberately avoids premature market-area or store-stage classification |
| Production deployment | Not implemented |
| Automated daily operating recommendations | Not implemented |
| Direct Qdrant loading path for Demo 2 facts | Not implemented; Demo 2 endpoint is currently file-backed |
| Retrieval endpoint for Demo 3 pairwise gate | Not implemented; Demo 3 is currently offline SQL / output / eval |

## Validation Commands

Current project checks are expected to pass with these commands:

- `python3 scripts/validate_demo2_api_endpoint.py`
- `python3 retail_ops/scripts/validate_demo2_staging_data.py`
- `python3 retail_ops/scripts/validate_demo2_comparability_output.py`
- `python3 retail_ops/scripts/validate_demo2_retail_memory_facts.py`
- `python3 eval/eval_retail_demo2_facts.py`
- `python3 eval/eval_retail_demo2_comparability_gate.py`
- `python3 eval/eval_retail_demo2_answer_behavior.py`
- `python3 retail_ops/scripts/run_demo3_pairwise_gate.py`
- `python3 retail_ops/scripts/validate_demo3_pairwise_gate_output.py`
- `python3 eval/eval_retail_demo3_pairwise_gate.py`
- `python3 retail_ops/scripts/validate_retail_data_contract.py`
- `python3 scripts/validate_project_consistency.py`

Expected retail evaluation status:

- Store A retail eval: Retail eval result: 8/8 passed
- Demo 2 offline facts eval: Retail Demo 2 facts eval result: 6/6 passed
- Demo 2 comparability-gate eval: Retail Demo 2 comparability-gate consistency eval result: 5/5 passed
- Demo 2 answer-boundary eval: Retail Demo 2 answer-behavior boundary eval result: 4/4 passed
- Demo 3 pairwise gate eval: Retail Demo 3 pairwise gate eval result: 9/9 passed

## Retail Demo 2 Answer-Behavior Boundary

Demo 2 includes an offline answer-boundary check:

- `eval/eval_retail_demo2_answer_behavior.py`
- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

This check focuses on whether comparison answers preserve the implemented metric contract:

- `activity_cost_ratio_pct` is treated as activity-cost-ratio evidence, not ROI or profit margin.
- `top3_sku_transaction_amount_share_pct` is treated as lightweight top-SKU concentration evidence, not full product-category sales share.
- search-entry comparison stays tied to `search_entry_rate_pct`, `search_entry_share_pct`, `search_entry_users`, and `entry_users`.
- promotion or subsidy strategy transfer is qualified unless activity, subsidy, refund, invalid-order, and comparison-limit evidence support it.

## Retail Demo 3 Pairwise Comparability Boundary

Demo 3 includes an offline pairwise-gate evaluation:

- `eval/eval_retail_demo3_pairwise_gate.py`
- `eval/results/eval_retail_demo3_pairwise_gate_result.txt`

This check focuses on whether pairwise comparison remains narrow:

- pairwise output has the expected three question types;
- activity-transfer comparison can refuse unsupported transfer;
- search-entry and order-quality comparisons stay within their question boundaries;
- `region_type` remains weak context instead of a hard grouping rule;
- pairwise output does not become best-store ranking;
- new pairwise fields are documented.
