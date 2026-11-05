# Candidate Brief

Refine the incumbent: `common_divisor_sharpened_exemplar` — rank 0, the best model on this data, ELPD-LOO -2744.1. Its hypothesis, as its author stated it:

> Refinement of the incumbent `coincidence_sharpened_exemplar`. People still judge a target by its summed similarity to the examples (magnitude, final digit, parity, and a factor dimension), and they still sharpen this generalization on every dimension when the examples share a suspicious coincidence. The one change is to the factor dimension. People do not check pairwise whether the target shares any factor with each example. Instead they read the common divisor of the whole set (when the examples share one of 3 or more, e.g. 3 in "3, 63" or 10 in "10, 50, 100"), and a target that is not a multiple of it loses similarity to every example by a constant factor. So multiples of the set's common divisor that match no example's final digit are endorsed well above non-multiples, while single examples and sets without such a divisor carry no factor penalty.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment2/model_loop/models/common_divisor_sharpened_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
