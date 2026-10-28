# Candidate Brief

Refine the incumbent: `graded_factor_coherence_tuned_exemplar` — rank 0, the best model on this data, ELPD-LOO -1551.7. Its hypothesis, as its author stated it:

> Refinement of graded_shared_factor_exemplar_similarity: people still judge a target by its recency-weighted exponential similarity to the remembered examples, with the number-theoretic part of similarity graded by how large a factor the target and each example share, but the attention they give each kind of similarity (size, final digit, parity, shared factor) is stretched toward the respects in which the examples resemble one another. The single change grafts this coherence-tuned attention (from coherence_tuned_attention_exemplar) onto the graded model, with the examples' factor coherence itself measured by the same graded shared-factor match, so sets whose examples share a large common factor make people generalize sharply to targets sharing that factor, closing the common-factor rule gap, while a single example leaves attention at baseline.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/graded_factor_coherence_tuned_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
