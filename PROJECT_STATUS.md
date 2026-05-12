# Project Status

This file records the current implementation status and project boundary.

Admissions-facing summary: `PROJECT_SUMMARY_FOR_ADMISSIONS.md`

Main technical overview: `README.md`

Retail operations evidence: `retail_ops/`

## 1. Current Implemented Scope

| Area | Status | Main evidence |
|---|---|---|
| Livestream memory layer | Implemented local prototype | `api/main.py`, `/chat_mem`, `/chat_livestream_kb`, `eval/eval_report.md` |
| Typed memory lifecycle | Implemented for livestream product facts | fact policy registry, overwrite control, soft deactivation, freshness filtering, active-state filtering |
| Retail data dictionary | Implemented | `retail_ops/data/DATA_DICTIONARY.md` |
| Retail lineage | Implemented | `retail_ops/LINEAGE.md` |
| Retail Demo 1 | Implemented | Store A month-over-month diagnostic |
| Retail Demo 2 | Implemented | Stores B-F same-period cross-store diagnostic |
| Retail Demo 3 | Implemented as offline SQL / output / evaluation plus narrow file-backed answer path | pairwise comparability gate and answer script over Demo 2 B-F output |
| Retail data-contract validation | Implemented | `retail_ops/scripts/validate_retail_data_contract.py` |
| Project consistency validation | Implemented | `scripts/validate_project_consistency.py` |

## 2. Current Retail Prototype

The retail extension currently has three staged demos.

| Demo | Scope | Main question |
|---|---|---|
| Demo 1 | Store A across February, March, and April 2026 | What changed inside one store across observed months? |
| Demo 2 | Stores B-F under the same March 2026 reporting window | How can selected stores be compared under the same period? |
| Demo 3 | Pairwise comparison over Demo 2 output | Can two stores be compared for one specific operating question? |

The current retail prototype is built around a comparability-first decision-support question:

Which store-period rows can be compared, under what conditions, and which conclusions should be limited or refused?

## 3. Current Boundary

This repository does not claim to be a full 48-store automated Meituan operations system.

### Not Yet Implemented

| Area | Current boundary |
|---|---|
| Automated Meituan backend ingestion | Not implemented |
| Full 48-store automated decision support | Not implemented |
| Production deployment | Not implemented |
| Automated daily operating recommendations | Not implemented |
| Automated SKU category classification across the full catalog | Not implemented |
| Causal attribution of performance change to one factor | Not implemented |
| Demo 3 retrieval endpoint | Not implemented; Demo 3 currently has a narrow file-backed answer path, not a retrieval endpoint or API endpoint |

## 4. Current Consistency Focus

The narrow Demo 3 answer path is already implemented as a file-backed script.

That answer path reads:

`retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`

It answers a specific pairwise question using:

| Input or output field | Purpose |
|---|---|
| `reference_store_id` | reference store |
| `candidate_store_id` | candidate store |
| `comparison_question_type` | narrow comparison question |
| `pairwise_comparison_decision` | gate decision |
| gap fields | supporting evidence |
| `pairwise_limit_notes` | comparison boundary |
| short grounded answer | readable decision-support response |

The current focus is consistency cleanup, not adding another demo. The project should continue to present Demo 3 as a narrow pairwise answer path over the existing B-F sample, not as a full 48-store operating platform, not as a final recommendation engine, and not as a broad market-classification system.

## 5. Validation Commands

Current project checks:

`git diff --check`

`python3 retail_ops/scripts/validate_retail_data_contract.py`

`python3 scripts/validate_project_consistency.py`

`python3 eval/eval_livestream.py`

`python3 eval/eval_retail.py`

`python3 eval/eval_retail_demo2_facts.py`

`python3 eval/eval_retail_demo2_comparability_gate.py`

`python3 eval/eval_retail_demo2_answer_behavior.py`

`python3 eval/eval_retail_demo3_pairwise_gate.py` `python3 eval/eval_retail_demo3_pairwise_answer_path.py`

If `eval/eval_livestream.py` fails because `httpx` is missing, that is a local Python dependency issue rather than a project consistency failure.

## 6. One-Line Status

This repository currently demonstrates lifecycle-aware product memory plus a staged Meituan-style retail decision-support prototype: Store A month-over-month diagnosis, B-F same-period diagnostic comparison, B-F pairwise comparability gating, and a narrow file-backed Demo 3 answer path.
