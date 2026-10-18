# Candidate Brief

Refine the incumbent: `mutual_similarity_contrast_standard` — rank 0, the best model on this data, ELPD-LOO -4736.4. Its hypothesis, as its author stated it:

> People judge a target's resemblance to the examples relative to how much the examples resemble one another on the same final-digit metric (same digit most alike, same parity partly, opposite parity least): the examples' mutual similarity sets the standard a target must meet. When the examples closely resemble each other (shared digits or parity), a target that matches none of their final digits falls well short of that standard and is rejected. When the examples share little (mixed digits and parities), even a dissimilar target is about as alike to them as they are to each other, so people accept broadly. Apart from this, judgments follow the current best model: rule-pre-empted parity credit, a common-factor boost, and rule-gated decade-band familiarity.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/mutual_similarity_contrast_standard.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
