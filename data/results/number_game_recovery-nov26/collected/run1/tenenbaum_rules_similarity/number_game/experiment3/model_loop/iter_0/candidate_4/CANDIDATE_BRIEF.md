# Candidate Brief

Refine the incumbent: `centre_weighted_range_sampling` — rank 0, the best model on this data, ELPD-LOO -3524.0. Its hypothesis, as its author stated it:

> Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over number rules and numeric ranges, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over range width, but they do not expect a range-based program to produce every number in its range equally often — they expect its outputs to cluster toward the middle of the range, with how strongly they do so inferred from the data (none recovers the parent model). The one change is this centre-weighted sampling from ranges: examples sitting at a range's edge are a "suspicious coincidence" for it, so ranges that are centred on the examples and extend beyond the lowest and highest example gain support, and people say "yes" to targets somewhat past the examples' extremes more often than a uniform-sampling learner predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/centre_weighted_range_sampling.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
