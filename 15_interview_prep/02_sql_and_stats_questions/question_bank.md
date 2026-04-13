# SQL And Stats Question Bank

## 1. How do `where` and `having` differ?

- Short answer: `where` filters rows before aggregation, while `having` filters groups after aggregation.
- Example: use `where event_date >= '2026-01-01'` before grouping, then `having count(*) > 10` after grouping.

## 2. When would you use a window function?

- Use it when you need row-level output with group context, such as running totals, ranking, retention, or rolling averages.

## 3. What is the difference between confidence interval and hypothesis test?

- A confidence interval gives a plausible range for the parameter.
- A hypothesis test evaluates whether the observed evidence is strong enough against a null hypothesis.

## 4. What usually breaks an A/B test?

- sample ratio mismatch
- leakage between variants
- stopping early without a rule
- changing targeting mid-experiment

## 5. Why can high accuracy still be misleading?

- Accuracy can hide class imbalance. Precision, recall, F1, PR curves, or cost-sensitive framing may matter more.
