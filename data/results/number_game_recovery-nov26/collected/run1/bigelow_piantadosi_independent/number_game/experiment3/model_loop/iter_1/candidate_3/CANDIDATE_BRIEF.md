# Candidate Brief

Refine the incumbent: `anchored_neighbourhood_side_channel` — rank 0, the best model on this data, ELPD-LOO -2638.9. Its hypothesis, as its author stated it:

> Refinement of `rule_with_exception_side_channel`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient) and moderate-width ranges on a compressed number line, and they picture a "rule plus exception" program the same way — a main rule channel that, at a fitted rate, emits a member of its exception instead — but the exception is a small fuzzy neighbourhood (radius zero to three, smaller radii a priori likelier) centred on an example the rule fails to cover, rather than an arbitrary single number or a whole decade. The one change is this example-anchored neighbourhood as the side channel's exception (grafted from `example_anchored_exception_neighbourhood`): because a wider neighbourhood only dilutes the off-pattern example's own likelihood, not the rule-following examples', people readily posit a broad region around the outlier and endorse its close neighbours (falling off with distance) much more than off-rule neighbours of the majority examples — more strongly than when rule and neighbourhood are sampled as one pooled set.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment3/model_loop/models/anchored_neighbourhood_side_channel.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
