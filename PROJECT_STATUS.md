# Project Status

This file records the current implementation boundary of the repository. It is intentionally short.

Admissions-facing narrative belongs in PROJECT_SUMMARY_FOR_ADMISSIONS.md. The technical overview belongs in README.md. Detailed retail evidence belongs in the retail demo, dictionary, lineage, SQL outputs, generated facts, and evaluation files.

## Current Implemented Scope

| Area | Status | Evidence |
|---|---:|---|
| Livestream memory layer | Implemented prototype | api/main.py, eval/eval_report.md, eval/results/eval_result_11_pass.txt |
| Typed memory and lifecycle-aware retrieval | Implemented prototype | fact policy registry, retrieval gating, stale / unsupported fallback behavior |
| Retail Demo 1 | Implemented Store A month-over-month diagnostic | retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md |
| Retail Demo 2 | Implemented limited B-F same-period cross-store comparability diagnostic | retail_ops/outputs/demo2_cross_store_comparability_output.csv |
| Retail source metric dictionary | Implemented | retail_ops/data/DATA_DICTIONARY.md |
| Retail claim-to-field lineage | Implemented for Demo 1 and Demo 2 | retail_ops/LINEAGE.md |
| Store A SQL diagnostic output | Implemented | retail_ops/sql/01_store_a_month_over_month_diagnostic.sql, retail_ops/outputs/store_a_demo1_sql_output.csv |
| Demo 2 SQL diagnostic output | Implemented | retail_ops/sql/02_demo2_cross_store_comparability.sql, retail_ops/outputs/demo2_cross_store_comparability_output.csv |
| Retail memory-facing facts | Implemented for Store A Demo 1 and Demo 2 B-F facts | retail_ops/outputs/generated_retail_memory_facts.json, retail_ops/outputs/generated_demo2_retail_memory_facts.json |
| Retail validation | Implemented | retail_ops/scripts/validate_retail_data_contract.py, Demo 2 validation scripts, scripts/validate_project_consistency.py |
| Retail retrieval evaluation | Implemented for Store A Demo 1; Demo 2 facts and comparability-gate consistency have offline evals | eval/eval_retail.py, eval/eval_retail_demo2_facts.py, eval/eval_retail_demo2_comparability_gate.py |
| Retail API endpoints | Implemented as separate Demo 1 and Demo 2 endpoints | /chat_retail_ops_kb, /chat_retail_ops_demo2_kb |

## Current Boundary

The retail extension now has two implemented stages.

Demo 1 demonstrates how Meituan backend metrics can be normalized, checked, traced, and converted into memory-facing facts for a Store A month-over-month diagnostic.

Demo 2 adds a limited same-period cross-store diagnostic for anonymized Stores B-F using March 2026 backend data. It structures cross-store source metrics, derives comparability diagnostics, generates Demo 2 retail memory facts, validates those facts, and exposes a separate file-backed Demo 2 API endpoint.

Demo 2 does not rank stores as simply better or worse. It is a comparability-first diagnostic, not a final operating recommendation system.

The current implementation does not yet perform full multi-store comparability grouping across all 48 stores.

## Not Yet Implemented

| Area | Current boundary |
|---|---|
| Automated Meituan backend ingestion | Not implemented |
| Full 48-store cross-store comparison | Not implemented |
| Store-stage classification across all stores | Not implemented |
| Production deployment | Not implemented |
| Automated daily operating recommendations | Not implemented |
| Direct Qdrant loading path for Demo 2 facts | Not implemented; Demo 2 endpoint is currently file-backed |

## Validation Commands

Current project checks are expected to pass with:

- python3 scripts/validate_demo2_api_endpoint.py
- python3 retail_ops/scripts/validate_demo2_staging_data.py
- python3 retail_ops/scripts/validate_demo2_comparability_output.py
- python3 retail_ops/scripts/validate_demo2_retail_memory_facts.py
- python3 eval/eval_retail_demo2_facts.py
- python3 eval/eval_retail_demo2_comparability_gate.py
- python3 retail_ops/scripts/validate_retail_data_contract.py
- python3 scripts/validate_project_consistency.py

Expected retail evaluation status:

- Store A retail eval: Retail eval result: 8/8 passed
- Demo 2 offline facts eval: Retail Demo 2 facts eval result: 6/6 passed
