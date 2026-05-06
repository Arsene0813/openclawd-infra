# Retail Decision Memory Architecture

This document explains how the original livestream memory layer is extended into a retail operations decision-support prototype.

## Original Problem

The original livestream commerce system focused on changing product information:

- price
- stock status
- promotion
- shipping policy
- product features

These facts change over time. If an AI assistant treats them as timeless chat history, it may reuse outdated information.

The memory layer addresses this problem by storing structured facts with type, entity, source, active state, freshness rules, and traceable retrieval.

## Extended Retail Operations Problem

Multi-store Meituan operations have a similar lifecycle problem.

Operational observations also change over time:

- traffic structure changes
- promotions expire
- refund pressure changes
- store rankings shift
- SKU roles vary by region and season
- fulfillment reliability affects conversion quality

Therefore, operational knowledge should also be structured, scoped, updated, and reused carefully.

## Architecture

```text
Livestream product interaction
        |
        v
Structured product memory
        |
        v
Lifecycle-aware retrieval
        |
        v
Traceable answer or safe fallback


Meituan backend metrics
        |
        v
Anonymized store-level CSV
        |
        v
SQL-derived operational metrics
        |
        v
Cross-store ranking and conservative tags
        |
        v
Structured operational memory facts
        |
        v
Evidence-based decision support or qualified refusal
