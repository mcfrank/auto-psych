# Candidate Brief

Refine the incumbent: `even_odd_split_family_prior` — rank 0, the best model on this data, ELPD-LOO -872.7. Its hypothesis, as its author stated it:

> Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, with a tempered size principle and a prior over kinds of rule, but they do not treat "even" and "odd" as one kind of rule with a single shared plausibility — "even numbers" and "odd numbers" are separate kinds, each with its own prior plausibility inferred from the data. The single change is splitting the parity family in two, predicting that an all-even set of examples triggers parity generalization (yeses to distant same-parity targets) more readily than an all-odd set does.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment1/model_loop/models/even_odd_split_family_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
