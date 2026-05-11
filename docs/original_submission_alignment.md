# Alignment with Submitted Supplementary Material

My submitted supplementary material described this project as an evolution from an earlier LLM-powered livestream system into a lifecycle-aware agent memory layer.

The current repository keeps the same core direction. The main components described in the submitted material are still reflected in the implementation and documentation:

| Submitted Material Theme | Repository Component |
|---|---|
| Structured memory | typed fact storage and fact-policy registry |
| Retrieval control | retrieval gating, fallback, and refusal behavior |
| Fact update handling | slot-based overwrite and soft deactivation |
| Memory freshness | timestamps, freshness windows, active-state flags |
| Traceability | supporting memory items returned with answers |
| Livestream commerce use case | product price, promotions, stock status, shipping policy, product features |

Later updates mainly improve clarity, evaluation coverage, and connection to broader commerce decision-support scenarios.

## Current Retail Extension Boundary

This file explains how the earlier submitted material maps to the original lifecycle-aware memory-layer prototype.

The current repository also extends the same evidence-control design into Meituan-style instant retail operations. The current admissions-facing narrative is summarized in `PROJECT_SUMMARY_FOR_ADMISSIONS.md`, while the retail implementation details are documented under `retail_ops/`.

The retail extension should be read as a staged decision-support prototype based on real store-operation problems, not as a finished automated operating system or a complete 48-store deployment.
