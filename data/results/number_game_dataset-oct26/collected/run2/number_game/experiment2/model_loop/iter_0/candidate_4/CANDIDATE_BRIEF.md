# Candidate Brief

Refine the incumbent: `dual_route_shared_factor_rule_exemplar` — rank 0, the best model on this data, ELPD-LOO -3120.5. Its hypothesis, as its author stated it:

> Refinement of dimension_specific_coherence_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor), with each dimension's attention stretched by its own sensitivity to how much the examples resemble one another, but when the examples all share a common factor of 3 or more they also abstract the explicit rule "multiples of that factor" and consult it alongside similarity when answering. The single change is this rule as a separate source of evidence at the decision, not a dimension of similarity: it raises "yes" for multiples and lowers it for non-multiples regardless of how far they are in size, in proportion to how unlikely the shared factor would be by coincidence (larger factors and more examples), so far-away multiples of g are endorsed and nearby non-multiples rejected, closing the common-factor rule gap that similarity alone leaves.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/dual_route_shared_factor_rule_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
