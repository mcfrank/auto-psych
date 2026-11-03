# Candidate Brief

Refine the incumbent: `gamma_shaped_range_width_prior` — rank 0, the best model on this data, ELPD-LOO -2622.7. Its hypothesis, as its author stated it:

> Refinement of `anchored_neighbourhood_side_channel`: people do the same tempered, exception-tolerant Bayesian concept learning (the same rules with salient final-digit rules, ranges sampled along a compressed number line, and rule-plus-neighbourhood exceptions anchored on an off-pattern example and emitted through a side channel), but their prior expectation of how wide a "range of numbers" program is does not have the fixed Erlang shape, which makes ranges of one or two numbers unlikely. Instead it is a gamma-shaped width prior whose shape is fitted, so it can favour tight ranges hugging an example while keeping a long tail of broad ones. The one change is this free shape of the range-width prior: if people find tight ranges plausible, they generalise from a lone example steeply to its close neighbours and drop off fast with distance, more than the incumbent allows (the critique's single-example distance gradient), while still extending broad rules to distant numbers.

Its source is `/scr/wurgaft/auto-psych/agent_trees/b1163c305e22f5d0/repo/_runs/cell_1/experiment3/model_loop/models/gamma_shaped_range_width_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
