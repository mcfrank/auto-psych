# Candidate Brief

Refine the incumbent: `without_replacement_size_principle` — rank 0, the best model on this data, ELPD-LOO -2256.8. Its hypothesis, as its author stated it:

> Refinement of `threshold_concepts_erlang_interval`: people do the same Bayesian concept learning over number rules, bounded ranges (with an Erlang prior on width) and one-sided thresholds, with a prior mass per concept family, but they believe the program never repeats itself — each example is a member of the concept not already shown, drawn without replacement. The one change is this without-replacement size principle, which makes the "suspicious coincidence" grow faster than an ideal with-replacement learner's whenever the examples use up a large share of a small concept: sets that nearly exhaust a tiny rule or a narrow range (e.g. 4 of the 7 powers of 2, or 11, 15 filling a five-number band) favour that tight concept even more strongly, so people reject targets just outside it more often than the best model predicts, while sets that are few relative to every candidate concept behave as before.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/without_replacement_size_principle.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
