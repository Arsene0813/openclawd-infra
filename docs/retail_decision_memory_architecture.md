# Retail Decision Memory Architecture

This document explains how the original livestream memory layer is extended into a retail operations decision-support prototype.

## 1. Original Livestream Memory Problem

The original system focused on changing product information in livestream or retail-commerce interaction.

Examples include:

- product price
- stock status
- promotion
- shipping policy
- product features

These facts change over time. If an AI assistant treats them as ordinary chat history, it may reuse outdated or conflicting information.

The memory layer addresses this problem by storing product-related information as structured facts with type, entity, slot, active state, update behavior, freshness rules, and traceable retrieval.

## 2. Extended Retail Operations Problem

Multi-store Meituan operations have a similar lifecycle problem.

Operational observations also change over time. For example:

- search traffic structure changes
- promotion effects expire
- refund pressure changes
- store rankings shift
- product roles vary by region and season
- fulfillment reliability affects conversion quality

Therefore, operational knowledge should also be structured, scoped, updated, checked, and reused carefully.

## 3. Architecture Overview

```text
Original livestream scenario:

Product interaction
        ↓
Structured product facts
        ↓
Lifecycle-aware memory
        ↓
Freshness-aware retrieval
        ↓
Traceable answer or safe fallback


Retail operations extension:

Meituan backend metrics
        ↓
Anonymized store-level data
        ↓
SQL-derived operational metrics
        ↓
Cross-store comparison
        ↓
Structured operational memory facts
        ↓
Evidence-based decision support or qualified refusal
```

## 4. Why SQL Is Added Before Memory

Raw store-level backend data is difficult to compare directly across stores.

SQL is used to calculate comparable indicators such as:

- search visit share
- search exposure-to-visit rate
- refund revenue ratio
- refund order ratio
- promotion GMV to revenue ratio
- merchant subsidy to revenue ratio
- revenue per visitor

Only after this structuring step should selected observations be converted into operational memory facts.

## 5. Operational Memory Fact Types

The retail extension can use memory fact types such as:

- `cross_store_pattern`
- `store_metric_signal`
- `risk_signal`
- `promotion_efficiency_signal`
- `traffic_efficiency_signal`
- `sku_role_signal`

These facts should include:

- period
- store scope
- metric source
- confidence level
- decision-support use
- limitation or caution

## 6. Conservative Retrieval Principle

The assistant should not treat operational memory as universal truth.

Before giving a recommendation, it should check:

- whether the data period is still relevant
- whether the store comparison is fair
- whether the observation is descriptive or causal
- whether the user is asking beyond the available evidence
- whether refund pressure, promotion cost, product mix, and fulfillment reliability have been considered

## 7. Example

A memory fact may say:

> Store D has strong revenue, order count, search exposure-to-visit rate, and visitor value in the March 2026 public sample.

A conservative assistant should not answer:

> Copy Store D's strategy to all stores.

A better answer is:

> Store D can be reviewed as a reference store, but its strategy should not be copied directly without checking region, product mix, promotion structure, fulfillment conditions, and customer profile.

## 8. Current Limitation

This is a prototype for data readiness and decision-memory design.

It does not claim to replace human business judgment, run full causal inference, forecast demand, or automatically optimize all stores.

The purpose is to show how fragmented store-level metrics can be standardized, compared, and converted into traceable memory facts before being reused by an AI assistant.
