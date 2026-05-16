# Document Responsibility Map

This file defines the responsibility of each reviewer-facing and retail-operations document.

The purpose is to reduce narrative overlap without removing evidence. Full explanations should live in the document that owns that topic. Other documents should summarize briefly and link to the owner document.

## Core Rule

Each concept should have one primary explanation location.

Other documents may mention the concept only as a short summary, navigation pointer, or boundary reminder.

## Responsibility Table

| Document | Primary Responsibility | Should Avoid |
|---|---|---|
| `README.md` | Repository landing page: project purpose, implemented scope, fast reading path, key checks, and current limitations. | Long demo explanations, full future-gate design, detailed metric definitions. |
| `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Admissions-facing narrative: real business problem, analytical shift, implemented prototype, and honest current boundary. | Long file lists, detailed formulas, script-by-script implementation notes. |
| `ADMISSIONS_REVIEW_GUIDE.md` | Reviewer navigation path for 30-second, 3-minute, and deeper review. | Repeating the full project story. |
| `PROJECT_STATUS.md` | Current implementation status, validation focus, and commands. | Business narrative and detailed metric explanation. |
| `retail_ops/README.md` | Folder map for the retail operations extension. | Repeating the full admissions narrative or future-gate specification. |
| `retail_ops/data/DATA_DICTIONARY.md` | Canonical source of truth for field names, metric definitions, data grain, and naming boundaries. | Admissions-style project storytelling. |
| `retail_ops/LINEAGE.md` | Claim-to-data lineage from source fields to SQL outputs, generated memory facts, and answer-boundary checks. | Repeating the full business problem or future-gate design. |
| `retail_ops/EXPERIMENTS.md` | Analytical checks, expected behavior, pass conditions, and failure modes. | Demo evidence interpretation or admissions narrative. |
| `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Store A month-over-month evidence and interpretation boundary. | Cross-store future-gate explanation. |
| `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | Stores B-F same-period diagnostic evidence and interpretation boundary. | Full future pairwise-gate specification. |
| `retail_ops/COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate design. | Claiming that the gate is currently implemented. |
| `eval/*.md` | Evaluation results and scenario-based behavior checks. | Repeating business narrative. |

## Repetition Control Rules

1. The full future pairwise comparability-gate design belongs in `retail_ops/COMPARABILITY_GATE_V0.md`.
2. Full metric definitions belong in `retail_ops/data/DATA_DICTIONARY.md`.
3. Full claim-to-field mapping belongs in `retail_ops/LINEAGE.md`.
4. Full experiment and validation logic belongs in `retail_ops/EXPERIMENTS.md`.
5. README files should guide readers, not duplicate every explanation.
6. Admissions-facing documents should explain value, not implementation details line by line.

## Safe Migration Principle

When reducing overlap, do not delete evidence-bearing content first.

Prefer this sequence:

1. Keep the full explanation in the owner document.
2. Replace repeated versions in other documents with a short summary.
3. Link to the owner document.
4. Run validation checks.
5. Commit the change.
