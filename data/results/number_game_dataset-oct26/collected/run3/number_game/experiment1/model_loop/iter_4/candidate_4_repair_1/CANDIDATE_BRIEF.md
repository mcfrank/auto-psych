# Candidate Brief

Refine the incumbent: `decade_aligned_range_concepts` — rank 0, the best model on this data, ELPD-LOO -1591.8. Its hypothesis, as its author stated it:

> Refinement of `sample_size_growing_evidence_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, with evidence weight growing with set size, but they think about number ranges in round decades, so a range whose edges fall on decade boundaries (such as 20-29 or 30-59) is a priori more plausible than an arbitrarily bounded one by an inferred bonus. The one change is this decade-aligned prior bonus on intervals, which predicts that a target just outside the examples' span is endorsed more when it lies in the same decade as an example (e.g. 29 after 23 and 25) than an equally near target across a decade boundary (e.g. 19), the decade-grouping discrepancy the critique of the best model flags.

Its source is `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/decade_aligned_range_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
