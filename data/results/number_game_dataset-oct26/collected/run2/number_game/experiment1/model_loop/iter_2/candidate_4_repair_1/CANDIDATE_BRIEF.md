# Candidate Brief

Refine the incumbent: `coherence_tuned_attention_exemplar` — rank 0, the best model on this data, ELPD-LOO -1551.9. Its hypothesis, as its author stated it:

> People judge a target by its recency-weighted similarity to the remembered examples, but how much attention they give each kind of similarity (closeness in size, shared final digit, shared parity, shared factor) is learned from the examples themselves: while reading the set they notice the respects in which the examples resemble one another and stretch attention toward those dimensions. So when all examples share a factor, final digit or parity, targets matching every example on that property are endorsed far more than pairwise similarity with fixed attention predicts, while a single example gives no such cue and leaves attention at its baseline.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/coherence_tuned_attention_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
