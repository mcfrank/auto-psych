# Candidate Brief

Refine the incumbent: `dual_route_factor_and_digit_rule_exemplar` — rank 0, the best model on this data, ELPD-LOO -3116.3. Its hypothesis, as its author stated it:

> Refinement of dual_route_shared_factor_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), and still consult an abstracted explicit rule alongside similarity at the decision, but the rule route is not limited to shared factors. The single change is that when all the examples end in the same digit, people also abstract the explicit rule "numbers ending in that digit" and consult it as separate evidence, raising "yes" for targets ending in that digit and lowering it for the rest regardless of size distance, in proportion to how unlikely that shared ending would be by coincidence (more examples), closing the "ends in d" gap the critique reports (people apply the final-digit rule more sharply than final-digit similarity predicts).

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/dual_route_factor_and_digit_rule_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
