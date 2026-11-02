# Candidate Brief

Refine the incumbent: `loose_range_sampling_sparse_rules` — rank 0, the best model on this data, ELPD-LOO -3516.0. Its hypothesis, as its author stated it:

> Refinement of `sparse_rule_familiarity_within_kinds`: people do the same capacity-limited Bayesian learning over rules (with sparse familiarity within each kind) and numerical ranges. The single change is that people do not apply the size principle equally to both kinds of explanation: they read the examples with one sampling strength when weighing arithmetic rules and with a separate, looser one when weighing stretches of the number line, so the examples count for less as evidence that a range is tight. This predicts that sets no rule fits are generalized somewhat beyond the examples' span, and that as examples accumulate, a fitting rule overtakes the tightest covering range faster than a single shared sampling strength allows.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/loose_range_sampling_sparse_rules.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
