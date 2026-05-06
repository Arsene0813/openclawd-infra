# Admissions Review Guide

This repository is intended to demonstrate my ability to connect AI system design, retrieval-augmented generation, structured memory, evaluation, SQL-based data analysis, and commerce decision-support problems.

## What to Review First

1. `README.md`  
Main project overview, current system capabilities, example workflows, running instructions, and the connection between livestream memory and retail decision support.

2. `retail_ops/README.md`  
A SQL-based extension showing how manually organized Meituan merchant backend metrics can be standardized into a cross-store comparison workflow.

3. `retail_ops/outputs/cross_store_comparison_report.md`  
A short report explaining the cross-store analysis of five anonymized stores, including search traffic structure, refund pressure, subsidy intensity, and visitor value.

4. `retail_ops/outputs/generated_memory_facts.json`  
A structured example showing how SQL outputs can be converted into operational memory facts with period, scope, decision use, confidence, and source.

5. `eval/eval_report.md`  
Scenario-based evaluation showing whether the memory layer can retrieve current facts, overwrite outdated facts, separate product entities, filter non-facts, and refuse unsupported queries.

6. `case_studies/from_livestream_to_retail_decision_support.md`  
A short case study explaining how the same memory architecture can be extended from livestream commerce to instant-retail operations and store-level decision support.

7. `PROJECT_SUMMARY_FOR_ADMISSIONS.md`  
A concise summary of the project motivation, problem, solution, example, and relevance to my intended studies.

## Why This Project Matters

A basic chatbot can generate fluent responses, but commercial decision-support systems need more than fluency. They need reliable memory, explicit update rules, freshness control, source traceability, evaluation, and careful reuse of operational observations.

This project explores how an LLM-based system can manage changing commercial knowledge such as product price, promotions, stock status, shipping policies, and product features.

The retail operations extension adds a second layer: SQL-based cross-store analysis. It shows how store-level backend metrics can be reorganized into comparable indicators before being converted into structured memory facts.

## Relevance to My Intended Programmes

This project is relevant to AI, computing, business analytics, and data science because it involves:

- structured information extraction
- retrieval-augmented generation
- typed memory design
- lifecycle-aware knowledge management
- SQL-based operational data analysis
- cross-store metric standardization
- evaluation of system behavior
- commerce-oriented decision support
- extension toward store-level operational analytics
- language-based search-intent understanding through short product queries
- mapping user/product queries and operational observations into structured facts for retrieval and decision support

## Current Evidence in This Repository

The repository currently includes:

- a working local prototype for lifecycle-aware product memory
- scenario-based evaluation with 11 / 11 current cases passed
- a SQL-based retail operations extension under `retail_ops/`
- cross-store Meituan metrics outputs for five anonymized stores
- conservative operational tags generated from SQL-derived metrics
- structured memory facts generated from retail operations observations

## Important Limitation

The retail operations extension is a small prototype, not a full automated business optimization system. Its purpose is to demonstrate how fragmented store-level metrics can be standardized, compared, and converted into traceable memory facts before being reused by an AI assistant.
