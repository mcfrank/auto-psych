# Candidate Brief

Refine the incumbent: `coherence_sharpened_decision_exemplar` — rank 0, the best model on this data, ELPD-LOO -1541.9. Its hypothesis, as its author stated it:

> Refinement of graded_factor_coherence_tuned_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with attention stretched toward the respects in which the examples resemble one another, but how much the examples resemble one another also sets how decisively people turn that similarity into an answer. The single change is that set coherence sharpens the decision rule: after a coherent set (examples alike in size, digit, parity or factors) people answer near-deterministically from similarity, endorsing similar targets and rejecting dissimilar ones almost always, whereas after an incoherent set or a single example their answers stay noisy, which makes set-level yes rates more extreme than the incumbent predicts (the between-set yes-rate spread in the critique) and sharpens the "ends in d" and "multiples of g" gaps for coherent sets.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/coherence_sharpened_decision_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
