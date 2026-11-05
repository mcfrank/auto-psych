# Candidate Brief

Refine the incumbent: `span_interval_similarity` — rank 0, the best model on this data, ELPD-LOO -4777.9. Its hypothesis, as its author stated it:

> When two or more examples are shown, people also treat the stretch of the number line between the smallest and largest example as a candidate concept in its own right ("numbers in this range"), and a target earns credit for it by its similarity to that interval: full credit anywhere inside it, decaying exponentially (Shepard's generalization gradient, at an inferred and fairly steep rate) with the target's distance outside it. This interval similarity enters, with its own salience, the salience-weighted share of the examples' shared profile that drives "yes" in the current best account, so targets lying between the examples are endorsed more than equally nondescript targets outside them, even when they share no digit, parity or seed-multiple with the examples.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/span_interval_similarity.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
