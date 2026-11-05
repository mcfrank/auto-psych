# Candidate Brief

Refine the incumbent: `parity_preserving_decisive_route` — rank 0, the best model on this data, ELPD-LOO -4123.9. Its hypothesis, as its author stated it:

> Refinement of `parity_preserving_multiple_route`: people still judge a target by its partly averaged similarity to the examples (magnitude, final digit, parity, the set's common divisor), with distance-free routes for factor-kin, whole-set round kinship and multiples of an example, the multiples route sharply weakened when the multiple has switched parity (34 or 86 after "17" or "43" rejected, 85 or 99 after "17" or "33" kept). The one change, grafted from `coincidence_decisive_exemplar`, is what a suspicious coincidence among the examples (all the same final digit, all multiples of 5, all the same parity) does: it no longer narrows the similarity gradient, it makes people more confident that a rule is at work, so their yes/no decision on that similarity becomes more all-or-none after coherent sets ("15, 85, 45", "12, 14, 16") and stays graded after single examples and incoherent sets.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/parity_preserving_decisive_route.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
