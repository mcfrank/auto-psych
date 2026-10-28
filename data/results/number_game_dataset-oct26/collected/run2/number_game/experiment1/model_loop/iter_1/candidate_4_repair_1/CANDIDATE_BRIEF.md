# Candidate Brief

Refine the incumbent: `recency_weighted_exemplar_similarity` — rank 0, the best model on this data, ELPD-LOO -1594.9. Its hypothesis, as its author stated it:

> People do not infer a hidden rule; they judge a target by its similarity to the remembered examples, as in an exemplar model. Similarity to each example decays exponentially with the psychological distance between the two numbers, a weighted sum of mismatches on how far apart they are, whether they share a final digit, whether they share parity, and whether one divides the other. Memory for the examples fades with how long ago each was shown, so the last-shown example counts most, and the probability of "yes" rises with the target's recency-weighted summed similarity to the examples.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment1/model_loop/models/recency_weighted_exemplar_similarity.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
