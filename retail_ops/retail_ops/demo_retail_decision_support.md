# Retail Decision-Support Demo

This demo shows how the retail operations SQL extension can be used together with lifecycle-aware memory principles.

The purpose is not to produce automatic business decisions. The purpose is to show how an AI assistant should use structured operational memory conservatively, with evidence, scope, and refusal rules.

## Demo Question 1

### User Question

Should I increase search exposure investment for all stores?

### Evidence Retrieved from Operational Memory

- In the March 2026 public sample, most stores are strongly search-driven.
- Four out of five stores have search visit share above 85%.
- However, store performance still differs across refund pressure, subsidy intensity, search exposure-to-visit rate, and revenue per visitor.

### Conservative System Answer

No. The current evidence does not support increasing search exposure investment for all stores equally.

Search visibility is important in this category, but it should not be treated as the only growth lever. Some stores may need refund review, promotion efficiency review, product mix adjustment, or fulfillment reliability checks before receiving additional traffic investment.

### Decision Principle

Search exposure can create opportunity, but it does not automatically create sustainable order growth.

---

## Demo Question 2

### User Question

Which store should be reviewed as a reference store?

### Evidence Retrieved from Operational Memory

- Store D has the highest revenue in the current public sample.
- Store D also shows strong visitor value and search exposure-to-visit performance.
- Store D has relatively low subsidy intensity compared with its revenue performance.

### Conservative System Answer

Store D can be reviewed as a reference store.

However, its operating pattern should not be copied directly to other stores without checking region, store type, product mix, promotion structure, fulfillment conditions, and customer profile.

### Decision Principle

A high-performing store can provide hypotheses, not universal rules.

---

## Demo Question 3

### User Question

Can I explain store growth only by search traffic?

### Evidence Retrieved from Operational Memory

- Search visit share is high across most stores.
- Store performance still differs across refund pressure, subsidy intensity, and revenue per visitor.
- Some stores with strong search-driven traffic still show refund or subsidy pressure.

### Conservative System Answer

No. Search traffic is an important entry source, but it does not fully explain store performance.

A more reliable analysis should compare traffic structure together with conversion quality, refund pressure, promotion cost, product mix, and fulfillment reliability.

### Decision Principle

Traffic metrics should be connected with quality and efficiency metrics before making operational decisions.

---

## Demo Question 4

### User Question

Which stores should be checked before increasing traffic investment?

### Evidence Retrieved from Operational Memory

- Stores with visible refund pressure should be reviewed before receiving more traffic.
- High refund revenue ratio or high refund order ratio may indicate product mismatch, unclear product descriptions, fulfillment problems, pricing issues, or promotion-related order quality problems.

### Conservative System Answer

Stores with refund pressure should be checked first.

Before increasing exposure, the operator should review refund reasons, product page clarity, SKU accuracy, pricing, delivery reliability, and after-sales issues.

### Decision Principle

Adding more traffic to a store with unresolved refund pressure may amplify operational problems.

---

## Demo Question 5

### User Question

Can this system automatically decide the best strategy for all Meituan stores?

### Evidence Retrieved from Operational Memory

- The public repository uses a small anonymized representative sample.
- The current workflow demonstrates data structuring, SQL comparison, conservative tagging, and memory-fact generation.
- It does not include complete causal testing, forecasting, or full operational automation.

### Conservative System Answer

No. The current system should not claim to automatically decide the best strategy for all stores.

It can support decision-making by organizing fragmented data, identifying comparable signals, generating traceable operational memory, and warning when evidence is insufficient.

### Decision Principle

The system is a decision-support prototype, not a fully automated business optimization system.
