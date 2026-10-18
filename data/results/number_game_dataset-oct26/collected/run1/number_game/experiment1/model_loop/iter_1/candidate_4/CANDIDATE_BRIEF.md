# Candidate Brief

Refine the incumbent: `exemplar_feature_similarity_gcm` — rank 0, the best model on this data, ELPD-LOO -1720.6. Its hypothesis, as its author stated it:

> People do not induce a rule; they judge the target by its exemplar similarity to the individual examples shown (as in the Generalized Context Model). Each number is represented on two psychological dimensions — its magnitude and its arithmetic features (parity, divisibility by 3/4/5/10, final digit, tens digit, being prime, square or a power of 2) — and similarity to each example decays exponentially with attention-weighted distance on these dimensions; the probability of "yes" grows with the summed similarity to the examples, so generalization is graded and does not sharpen with the size principle.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment1/model_loop/models/exemplar_feature_similarity_gcm.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
