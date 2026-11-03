# Candidate Brief

Refine the incumbent: `parity_veto_sparse_familiar_rules` — rank 0, the best model on this data, ELPD-LOO -3526.1. Its hypothesis, as its author stated it:

> Refinement of `parity_conflict_decision_veto`: people infer the hidden concept by capacity-limited Bayesian learning over rules (multiples of 5 as its own base-ten kind) and numerical ranges, and a target whose parity matches none of the examples triggers an automatic parity conflict that, on a fixed fraction of such trials, overrides the inferred concept with a "no". The single change is grafting sparse rule familiarity (from `sparse_rule_familiarity_within_kinds`) onto the concept-learning stage: most rules share their kind's plausibility, and only a few stand out as exceptionally familiar (multiples of 11, powers of 10) or unfamiliar (multiples of 7, "contains the digit 1"). The claim is that the decision-stage parity veto survives once rule priors are calibrated this way: opposite-parity targets in all-odd or all-even sets (e.g. odd numbers inside the span of 20, 32, 92) get fewer yeses than the inferred concept implies, while same-parity targets are unaffected.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/parity_veto_sparse_familiar_rules.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
