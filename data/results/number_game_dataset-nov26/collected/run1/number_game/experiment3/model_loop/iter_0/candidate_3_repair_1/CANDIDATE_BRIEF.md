# Candidate Brief

Refine the incumbent: `round_number_averaging_exemplar` — rank 0, the best model on this data, ELPD-LOO -4192.6. Its hypothesis, as its author stated it:

> Refinement of the incumbent `round_number_kinship_exemplar`, with one component grafted from `similarity_averaging_exemplar` (as used in `times_table_averaging_exemplar`): people pool a target's support across the examples by partly averaging rather than summing it. That support is the target's similarity to each example (magnitude, final digit, parity, the set's common divisor, sharpened by suspicious coincidences) plus the distance-free routes for multiples, factor-kin and round-number kinship. So each further example dilutes the support instead of adding to it. A far target in a multi-example set (e.g. "25, 82", "50, 2") is not carried by several weak resemblances added together, while lone examples ("7", "5") keep generalizing broadly to their multiples and round kin (14, 49, 98; 30, 100).

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/round_number_averaging_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
