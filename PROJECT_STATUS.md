# Project Status

This file records the current implementation boundary of the repository. It is intentionally short.

Admissions-facing narrative belongs in `PROJECT_SUMMARY_FOR_ADMISSIONS.md`. The technical overview belongs in `README.md`. Detailed retail evidence belongs in the retail demo, dictionary, lineage, SQL output, and evaluation files.

## Current Implemented Scope

| Area | Status | Evidence |
|---|---:|---|
| Livestream memory layer | Implemented prototype | `api/main.py`, `eval/eval_report.md`, `eval/results/eval_result_11_pass.txt` |
| Typed memory and lifecycle-aware retrieval | Implemented prototype | fact policy registry, retrieval gating, stale / unsupported fallback behavior |
| Retail operations extension | Implemented as Demo 1 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` |
| Retail source metric dictionary | Implemented | `retail_ops/data/DATA_DICTIONARY.md` |
| Retail claim-to-field lineage | Implemented | `retail_ops/LINEAGE.md` |
| Store A SQL diagnostic output | Implemented | `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`, `retail_ops/outputs/store_a_demo1_sql_output.csv` |
| Retail memory-facing facts | Implemented for Store A Demo 1 | `retail_ops/outputs/generated_retail_memory_facts.json` |
| Retail validation | Implemented | `retail_ops/scripts/validate_retail_data_contract.py` |
| Retail retrieval evaluation | Implemented | `eval/eval_retail.py`, `eval/eval_retail_report.md`, `eval/results/eval_retail_result.txt` |

## Current Boundary

The retail extension currently demonstrates how Meituan backend metrics can be normalized, checked, traced, and converted into memory-facing facts for a Store A month-over-month diagnostic.

The current implementation does not yet perform full multi-store comparability grouping across all 48 stores. The intended next step is a comparability-first layer that decides whether stores can be compared before generating cross-store operational interpretation.

## Not Yet Implemented

| Area | Current boundary |
|---|---|
| Automated Meituan backend ingestion | Not implemented |
| Full 48-store cross-store comparison | Not implemented |
| Store-stage classification across all stores | Not implemented |
| Production deployment | Not implemented |
| Automated daily operating recommendations | Not implemented |

## Validation Commands

Current project checks are expected to pass with:

```bash
git diff --check
python retail_ops/scripts/validate_retail_data_contract.py
python scripts/validate_project_consistency.py
python eval/eval_retail.py
```

Expected retail evaluation status:

```text
Retail eval result: 8/8 passed
```
