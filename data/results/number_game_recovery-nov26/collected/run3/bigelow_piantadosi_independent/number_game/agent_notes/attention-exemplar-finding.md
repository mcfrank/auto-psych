---
name: attention-exemplar-finding
description: "exp2 diagnostic-attention exemplar model, best MAP loglik about -2683 vs incumbent -2151; exemplar family is ~500 nats behind"
metadata:
  node_type: memory
  type: project
  originSessionId: fa66b65c-d764-4e20-b172-98d51405c12a
  modified: 2026-10-31T00:54:05.901Z
---

2026-10-30, cell_1/experiment2 iter_2/candidate_0 (`diagnostic_attention_exemplar_similarity`, exploratory exemplar brief). Incumbent hundred_chart_row_ranges MAP loglik -2151.3. The base-rate-only model gets -2943.
- The exemplar model uses property dimensions with attention set by the surprisal of the examples' agreement, plus magnitude. Mean similarity, ratio rule: -2791. Geometric-mean and max combinations, and squared agreement counts, gave about the same.
- Adding contains-digit dimensions: -2726. A logistic response on log S: -2707. Magnitude attention log(100/range): -2683 (submitted). Normalising attention to sum to 1 made it worse (-2723).
- Predictions stay too flat. There is no size principle across competing concepts, so it cannot commit.

**Why:** don't spend more time tuning exemplar/attention forms in this task.
**How to apply:** for exemplar briefs, expect large losses. Also, map.py needs the replace_rvs_by_values fix ([[factor-graded-multiples-finding]]).
