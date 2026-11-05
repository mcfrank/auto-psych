# Candidate Brief

Refine the incumbent: `coincidence_decisive_exemplar` — rank 0, the best model on this data, ELPD-LOO -4139.9. Its hypothesis, as its author stated it:

> Refinement of `parity_salience_asymmetry_exemplar` (partly averaged similarity to the examples on magnitude, final digit, asymmetrically salient parity and the set's common divisor, plus distance-free multiples, factor-kin and round-number routes). The one change is in what a suspicious coincidence among the examples (all the same final digit, all multiples of 5, all the same parity) does. It does not make people's similarity gradient narrower. Instead it makes them more confident that a rule is at work, so their yes/no decision becomes more decisive. Similarity still sets which targets are favoured, but after coherent sets ("15, 85, 45", "78, 48, 88", "12, 14, 16") people answer all-or-none: well-fitting targets are almost always accepted and poorly fitting ones almost always rejected. After single examples and incoherent sets, answers stay graded.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d82b8b9c1e886668/repo/_runs/cell_1/experiment3/model_loop/models/coincidence_decisive_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
