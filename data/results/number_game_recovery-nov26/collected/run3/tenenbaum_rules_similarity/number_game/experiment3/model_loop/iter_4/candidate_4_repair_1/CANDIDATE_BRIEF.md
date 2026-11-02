# Candidate Brief

Refine the incumbent: `rule_family_doubt_criterion` — rank 0, the best model on this data, ELPD-LOO -3662.9. Its hypothesis, as its author stated it:

> Refinement of `kind_uncertainty_shifted_criterion`: people infer the hidden concept exactly as its Bayesian concept learner does and shift their yes/no criterion with how unsure they are about what kind of program they face. The one change is which part of that kind-level doubt moves the criterion: only doubt about which kind of arithmetic rule it is (parity, primes, powers, multiples, final digit, ...), weighted by how much they believe it is a rule at all, shifts answers toward "no" (or "yes"), while doubt about whether the program is a rule or a stretch of the number line does not — the complement of `rule_vs_range_doubt_criterion`, which kept only that rule-versus-range part.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/rule_family_doubt_criterion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
