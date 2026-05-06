# Retail Operations SQL Extension

This folder extends the original livestream-agent memory layer into a retail operations decision-support scenario.

The original project focused on lifecycle-aware memory for livestream commerce facts such as price, stock status, promotion, shipping policy, and product features. This extension applies the same design principle to multi-store Meituan instant-retail operations: operational observations should not be treated as timeless memory. They should be structured, scoped, updated, checked, and reused only when the supporting data is still reliable.

## Motivation

Meituan merchant backend provides many useful store-level metrics, but the interface is mainly store-centric. For multi-store operations, the challenge is not simply whether data exists. The challenge is how to compare stores under the same metric structure before making decisions.

This extension demonstrates a small SQL-based workflow for reorganizing manually collected and anonymized Meituan backend metrics into a cross-store comparison layer. The goal is not to claim that the system can fully automate business decisions. The goal is to show how fragmented store-level metrics can be converted into traceable operational memory for conservative AI-assisted decision support.

## Data Scope

Public demo scope:

- Period: March 2026
- Stores: 5 anonymized representative stores
- Regions: Qingdao urban area and Yantai urban area
- Store types: self-operated and partner-operated
- Category: contact lenses and care-solution related instant retail
- Source: manually organized Meituan merchant backend metrics

Privacy note:

The public repository uses a small anonymized representative sample for readability and privacy protection. It should not be interpreted as the full operational scale of the underlying business context. The broader application supplement provides additional real-world operational background.

## Core Workflow

```text
Meituan backend metrics
        ↓
Anonymized store-level CSV
        ↓
SQLite / SQL derived metrics
        ↓
Cross-store ranking and conservative tags
        ↓
Structured operational memory facts
        ↓
Decision-support demo with evidence and refusal rules
