# Candidate Brief

Refine the incumbent: `drifting_criterion_dual_route_exemplar` — rank 0, the best model on this data, ELPD-LOO -3112.5. Its hypothesis, as its author stated it:

> Refinement of dual_route_factor_and_digit_rule_exemplar: people still judge a target by its recency-weighted exponential similarity to the remembered examples (size, final digit, parity, graded shared factor, each dimension's attention stretched by how much the examples resemble one another), and they still weigh the abstracted "multiples of g" and "ends in d" rules as separate evidence when they decide. The single change is that the criterion for saying "yes" relaxes steadily over the session: as people answer more and more questions without feedback, they need less evidence before they endorse a target. The same evidence therefore earns more "yes" answers late in the session than early on. This closes the late-minus-early yes-rate drift the critique reports, which a stationary model cannot produce.

Its source is `/scr/wurgaft/auto-psych/agent_trees/7e8e4116c4d40231/repo/_runs/cell_1/experiment2/model_loop/models/drifting_criterion_dual_route_exemplar.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
