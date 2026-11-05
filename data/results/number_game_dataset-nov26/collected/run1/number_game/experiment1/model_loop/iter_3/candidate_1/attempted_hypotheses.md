# Tried before

1 hypothesis proposed earlier in this project is no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### variability_tuned_parity_attention_2 — rejected (experiment1 round 2 candidate 4 refine incumbent variability_tuned_exemplar_attention)

**Outcome:** predicts like existing model 'parity_tuned_exemplar_attention' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of parity_tuned_exemplar_attention, not a new hypothesis.

**Hypothesis:** Refinement of `variability_tuned_exemplar_attention`: people still judge a target by its summed similarity to the remembered examples, with attention to magnitude and final digit set by how variable the examples are on each. The one change is that the same variability-driven attention also reaches parity (odd/even): the more the examples agree in parity (e.g. all even), the more a parity mismatch with an example lowers similarity, so after same-parity sets, same-parity targets far from every example and sharing no final digit are still endorsed more than opposite-parity ones, while mixed-parity sets ignore parity.
