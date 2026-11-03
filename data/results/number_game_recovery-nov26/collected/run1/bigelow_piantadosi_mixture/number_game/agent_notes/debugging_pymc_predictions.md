---
name: debugging-pymc-predictions
description: model.compile_fn on RV outputs samples random parameters; and how to make interval-sum models fast
metadata:
  type: reference
---
Pitfall: `model.compile_fn(model["p_yes"], inputs=model.value_vars)` silently draws the RVs at random, so p_yes looks inconsistent with a brute-force check. Use `pytensor.function(model.value_vars, model.replace_rvs_by_values([model["p_yes"]])[0], on_unused_input="ignore")(**ip)`.

Speed (iter_4 candidate_1, fan_retrieval_member_similarity, 2026-11-01): summing over all intervals with a parameter-dependent per-interval term can be done in closed form (the examples an interval holds form a contiguous run, so placements form a lo-range, and edge-distance similarity is a geometric series). Map compact valid (set, run, width) cells to pairs with a constant CSR (pm.Data for data/indices/indptr/shape, `pytensor.sparse.CSR` + `structured_dot`) and normalise with a scatter-add instead of padded softmax: gradient fell 70 ms -> 9 ms; smoke fit ~40 s. See [[check-candidate-runtime]].
