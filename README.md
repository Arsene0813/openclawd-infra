# Lifecycle-Aware AI Memory Layer for Retail Decision Support

Repository: `livestream-agent-memory-layer`

This is a local working prototype built from a real Meituan instant-retail operating problem.

The merchant backend provides detailed single-store metrics, but it does not naturally answer cross-store decision questions. Once the operation expands across many stores, the harder problem is not collecting more metrics. The harder problem is deciding which stores can be compared, under what conditions they can be compared, and what kind of operating judgment the evidence can support.

This project turns that problem into a staged decision-support prototype:

1. preserve backend metric definitions;
2. use SQL to structure selected store-period data;
3. convert diagnostic evidence into memory facts with source fields, observed values, source paths, and limitations;
4. test whether later answers stay inside the evidence boundary.

The goal is not to let an LLM make operating decisions directly. The goal is to reduce unsupported comparisons and preserve the evidence behind each answer.

---

## Fast Reading Path

For an admissions or project review, read in this order:

1. `PROJECT_SUMMARY_FOR_ADMISSIONS.md`
   Short admissions-facing explanation of the project and why it matters.

2. `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`
   Current retail Demo 2: same-period B-F cross-store diagnostic.

3. `retail_ops/data/DATA_DICTIONARY.md`
   Canonical Meituan-style backend metric definitions and field naming rules.

4. `retail_ops/LINEAGE.md`
   Source-to-SQL-to-memory lineage and interpretation limits.

5. `retail_ops/EXPERIMENTS.md`
   Current analytical experiment map and failure modes.

For implementation details, see `retail_ops/README.md`, `PROJECT_STATUS.md`, SQL files, scripts, generated outputs, and evaluation files.

---

## Current Scope

| Area | Implemented now | Not claimed |
|---|---|---|
| Livestream memory layer | Typed product facts, overwrite control, soft deactivation, active-state retrieval, fallback/refusal, scenario evaluation | General-purpose production agent memory platform |
| Retail metric dictionary | Meituan-style backend metric definitions and naming boundaries | Automatic ingestion from Meituan backend |
| Retail Demo 1 | Store A month-over-month diagnostic across February, March, and April 2026 | Causal explanation of monthly performance |
| Retail Demo 2 | Same-period B-F diagnostic with scope and limitation checks | Finished pairwise comparability gate, store ranking, or automatic strategy recommendation |
| Retail memory facts | Generated facts with observed values, source fields, source paths, supporting source paths, confidence, and limitations | Full 48-store automated decision system |
| Evaluation | Scenario-based checks for supported answers, unsupported-scope refusal, and metric-boundary preservation | Broad LLM benchmark |

The current implemented retail scope stops at Demo 2. A pairwise comparability gate is future work.

---

## Architecture

The prototype has two connected layers.

| Layer | Purpose | Main files |
|---|---|---|
| Memory-layer prototype | Store and retrieve typed facts while handling updates, stale knowledge, and unsupported questions | `api/`, `scripts/`, `eval/` |
| Retail operations extension | Structure Meituan-style backend metrics and preserve diagnostic evidence for cautious comparison | `retail_ops/` |

Basic flow:

```text
backend metrics / operator input
-> metric dictionary and data contract
-> SQL diagnostic output
-> generated memory facts
-> retrieval with source fields and limitations
-> qualified answer or refusal
```

The important design choice is that memory facts are not just summaries. They carry source fields, observed values, calculation notes, source paths, supporting source paths, confidence labels, and limitations.

---

## Retail Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes one self-operated Qingdao store across February, March, and April 2026.

It shows why operational performance should not be interpreted from one metric alone. Traffic, ranking, transaction amount, order conversion, activity cost, refund pressure, invalid orders, and top-SKU evidence can move in different directions.

Main file:

```text
retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md
```

The purpose is not to label the store as simply good or bad. The purpose is to keep month-over-month interpretation inside a documented evidence boundary.

---

## Retail Demo 2: Same-Period B-F Diagnostic

Demo 2 analyzes five anonymized stores, B-F, over the same March 2026 reporting window.

In this demo, comparability means row-level same-period diagnostic readiness. It does not mean pairwise store matching, store ranking, or strategy-transfer approval.

Demo 2 includes:

- same-period store-period metrics;
- top search-term evidence;
- top-SKU transaction-amount evidence;
- SQL-derived diagnostic fields;
- `comparison_scope_flag`;
- `comparison_limit_notes`;
- generated Demo 2 retail memory facts;
- facts evaluation and answer-boundary evaluation.

Main file:

```text
retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md
```

The future pairwise comparability gate should build on this structure, but it is not currently implemented as a finished demo.

---

## Evaluation Snapshot

The evaluations are scenario-based behavior checks. Their value is not proving that the model is generally correct. Their value is checking whether answers preserve metric definitions, source boundaries, and comparison limits.

| Check | Scope | Result |
|---|---|---:|
| Livestream memory evaluation | fact retrieval, overwrite behavior, entity separation, fallback/refusal, non-fact filtering | current implemented cases pass |
| Retail retrieval evaluation | Store A retail-memory retrieval and unsupported-scope refusal | 8/8 passed |
| Retail Demo 2 facts evaluation | Store B-F generated fact coverage across diagnostic slots | 6/6 passed |
| Retail Demo 2 answer-boundary evaluation | activity-cost ratio, top-SKU share, search-entry comparison, promotion-transfer limits | 4/4 passed |
| Retail data-contract validation | dictionary phrases, source/output headers, forbidden aliases, generated fact structure | passed |
| Project consistency validation | current-scope files, Demo 2 boundary wording, stale future-work artifacts, endpoint claims | passed |

---

## Reproduce Key Checks

Retail data-contract validation:

```bash
python3 retail_ops/scripts/validate_retail_data_contract.py
```

Demo 2 fact coverage evaluation:

```bash
python3 eval/eval_retail_demo2_facts.py
```

Demo 2 answer-boundary evaluation:

```bash
python3 eval/eval_retail_demo2_answer_behavior.py
```

Project consistency validation:

```bash
python3 scripts/validate_project_consistency.py
```

For the full local API setup with Docker Compose, Qdrant, Ollama, and FastAPI, see `PROJECT_STATUS.md` and `docker-compose.yml`.

---

## Key Evidence Files

| File | Why it matters |
|---|---|
| `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Admissions-facing summary of the real business problem and prototype scope |
| `retail_ops/data/DATA_DICTIONARY.md` | Canonical backend metric definitions and naming boundaries |
| `retail_ops/LINEAGE.md` | How source fields support SQL diagnostics and memory facts |
| `retail_ops/FIELD_USAGE_REVIEW.md` | Field-name review before future comparability-gate expansion |
| `retail_ops/EXPERIMENTS.md` | Experiment questions, inputs, transformations, pass conditions, and failure modes |
| `retail_ops/COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate design note |
| `retail_ops/sql/` | SQL transformations for Demo 1 and Demo 2 |
| `retail_ops/outputs/` | Generated diagnostic outputs and generated memory facts |
| `eval/` | Scenario-based evaluation scripts and reports |

---

## What This Demonstrates

This project demonstrates:

- turning a real retail operating problem into a structured data problem;
- preserving exact backend metric definitions instead of treating them as generic business metrics;
- using SQL to make selected store-period records comparable at the diagnostic level;
- converting diagnostic outputs into retrieval-facing memory facts;
- preserving source fields, observed values, source paths, supporting source paths, confidence, and limitations;
- testing whether answers qualify or refuse unsupported operating claims;
- defining future comparability requirements before strategy transfer.

The strongest point of the project is not that it is a finished operating platform. The strongest point is that it makes the evidence boundary explicit before comparing stores or suggesting operating actions.

---

## Current Limitations

Current limitations:

- Demo 1 covers Store A month-over-month analysis.
- Demo 2 covers a limited same-period B-F diagnostic.
- Automated Meituan backend ingestion is not implemented.
- Full 48-store peer selection and daily automated recommendations are not implemented.
- Promotion cycle dates, competitor density, delivery conditions, rating/review signals, and stockout history are not yet included.
- Estimated income is treated as a platform-displayed proxy, not audited profit.
- Top-SKU evidence is used as lightweight product-mix evidence, not full product-category sales share.
- `region_type` is weak context only, not a hard market-area classification.

## Future Work: Comparability Gate

Future work:

- build a pairwise comparability gate;
- decide whether two stores are comparable, comparable with limits, not comparable, or insufficiently supported;
- include order volume, transaction amount, activity status, activity intensity, store type, market context, competition, SKU structure, refund pressure, invalid-order pressure, and repeated reporting windows;
- avoid subjective regional classification until broader store data and stronger market-context evidence are available;
- preserve the same field contract and evaluation checks when expanding beyond the current sample.
