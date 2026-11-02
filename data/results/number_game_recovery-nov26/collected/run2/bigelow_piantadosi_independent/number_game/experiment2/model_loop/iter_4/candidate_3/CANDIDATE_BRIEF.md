# Candidate Brief

Refine the incumbent: `rule_familiarity_within_kinds` — rank 0, the best model on this data, ELPD-LOO -2055.6. Its hypothesis, as its author stated it:

> Refinement of `base_ten_multiples_capacity_limited`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with a prior plausibility for each kind of rule (odd, even, primes, squares/cubes, powers, multiples, final digit, contained digit, and multiples of 5 as its own base-ten kind), but the rules within a kind are not equally familiar — each rule carries its own familiarity that raises or lowers its prior plausibility relative to the other rules of its kind (a substantial, fixed amount of variation, with each rule's familiarity inferred from the data). The single change is replacing the equal sharing of a kind's plausibility among its rules with these rule-specific familiarities, predicting that some rules (e.g. multiples of 7, or "contains the digit 1") are rarely entertained even when the examples fit them, so those sets are generalized by range rather than by the rule, while other rules of the same kind (e.g. powers of 10, multiples of 11) are adopted readily.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/rule_familiarity_within_kinds.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
