# Demo 3: Pairwise Answer Path

## 1. Purpose

This file documents the narrow file-backed answer path for Demo 3.

The answer path reads the saved Demo 3 pairwise gate output and answers whether two stores can be compared for one supported question type.

It is intentionally small.

It is not a retrieval endpoint, not a ranking system, and not a final recommendation engine.

## 2. Source File

The answer path reads:

- `retail_ops/outputs/demo3_pairwise_comparability_gate_output.csv`

That output is generated from:

- `retail_ops/scripts/run_demo3_pairwise_gate.py`
- `retail_ops/sql/03_demo2_pairwise_comparability_gate.sql`

## 3. Script

The answer script is:

- `retail_ops/scripts/answer_demo3_pairwise_gate.py`

Example question:

- `Can Store B and Store E be compared for activity_transfer?`

The script returns:

- the stored pair;
- the requested pair;
- `comparison_question_type`;
- `pairwise_comparison_decision`;
- relevant gap fields;
- `pairwise_limit_notes`;
- a boundary note explaining that this is not a final operating recommendation.

## 4. Supported Question Types

Current supported `comparison_question_type` values are:

- `search_entry_structure`
- `activity_transfer`
- `order_quality_pressure`

The script should refuse or ask for a supported question type when the question does not match these implemented types.

## 5. Boundary Behavior

The answer path should refuse broad requests such as:

- full 48-store ranking;
- best-store selection;
- market-area classification;
- causal promotion-effect analysis;
- direct final operating recommendation.

The answer path should preserve the same boundary used by the Demo 3 pairwise gate:

- comparison first;
- limitation notes preserved;
- no direct strategy transfer;
- no claim that one store is better;
- no claim that activity caused the result.

## 6. Evaluation

The evaluation file is:

- `eval/eval_retail_demo3_pairwise_answer_path.py`

The current evaluation checks that the answer path:

- answers supported pairwise questions;
- preserves the comparability-gate boundary;
- includes relevant gap fields;
- includes limitation notes;
- refuses full 48-store ranking;
- reports missing store pairs;
- reports missing supported question types.

