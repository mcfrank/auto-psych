# Candidate Brief

Refine the incumbent: `range_size_prior_concepts` — rank 0, the best model on this data, ELPD-LOO -3157.5. Its hypothesis, as its author stated it:

> Refinement of `range_bounded_parity_concepts`: people do the same tempered-evidence Bayesian concept learning over the 45 weighted rules, all intervals of 1-100, their odd-only and even-only versions, and range-bounded rules, but they do not think every magnitude range is equally plausible a priori — a tiny range such as "numbers from 10 to 12" seems an unlikely thing for a program to produce, so a range's prior plausibility grows (or shrinks) as an inferred power of its size (power 0 recovers the incumbent). The one change is this size-dependent prior on ranges; because the size principle favours tiny ranges most when there is only one example, a broad-range prior moves belief from tiny ranges around a lone example to the rules it belongs to, so people generalize from a single example (10 → 80, 11 → 55) much more than the incumbent allows, while several examples still pin down a tight concept.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/range_size_prior_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
