# Candidate Brief

Refine the incumbent: `digit_rules_need_comparison` — rank 0, the best model on this data, ELPD-LOO -3132.2. Its hypothesis, as its author stated it:

> Refinement of `range_prior_sample_size_neglect` (the same tempered, sample-size-neglecting Bayesian concept learning over the 45 weighted rules, size-weighted intervals of 1-100, their odd/even versions and range-bounded rules). The one change: a digit pattern ("ends in 6", "contains a 1") is something people notice only by comparing examples, so when a single example is shown, digit-based rules hardly come to mind, and their prior is discounted by an inferred amount (zero recovers the incumbent). Belief then moves to the lone example's arithmetic rules. People should generalize a lone example to its multiples (6 → 12, 18; 11 → 55, 66; 10 → 80) more than the incumbent allows, and endorse numbers that share only its digits (14 → 74, 94) less. Judgments about sets of two or more examples are unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment2/model_loop/models/digit_rules_need_comparison.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
