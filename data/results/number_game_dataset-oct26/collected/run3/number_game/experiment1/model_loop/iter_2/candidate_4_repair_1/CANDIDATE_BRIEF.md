# Candidate Brief

Refine the incumbent: `sample_size_growing_evidence_concepts` — rank 0, the best model on this data, ELPD-LOO -1603.4. Its hypothesis, as its author stated it:

> Refinement of `conservative_evidence_interval_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted intervals of 1-100, but how much weight each example's evidence receives grows with the number of examples shown, because a single example gives little reason to believe it was sampled from a narrow concept, while several examples that all fit a tight concept convince people the examples are "strong" samples. The one change is that the evidence-weighting strength is a power function of the set size (an inferred growth exponent) rather than one constant, which predicts broader generalization from a single example than the incumbent, the discrepancy its critique flags.

Its source is `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment1/model_loop/models/sample_size_growing_evidence_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
