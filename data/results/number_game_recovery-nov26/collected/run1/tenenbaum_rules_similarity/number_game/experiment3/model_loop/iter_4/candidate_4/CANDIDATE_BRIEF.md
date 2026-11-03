# Candidate Brief

Refine the incumbent: `last_example_recency_weight` — rank 0, the best model on this data, ELPD-LOO -3517.8. Its hypothesis, as its author stated it:

> People infer the hidden program as a Bayesian concept learner over familiar number rules (all equally plausible), digit-sum rules and numeric ranges of a typical width, with the size principle, but they read the examples in the order shown and the most recently seen example stays most vivid: it carries its own evidential weight (more than an earlier example if recency dominates, less if it is discounted), while the earlier examples count normally. So for multi-example sets, which concepts survive depends on which example came last (16, 8, 64 is not read exactly like 64, 8, 16), while single-example sets behave as in the ideal learner.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment3/model_loop/models/last_example_recency_weight.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
