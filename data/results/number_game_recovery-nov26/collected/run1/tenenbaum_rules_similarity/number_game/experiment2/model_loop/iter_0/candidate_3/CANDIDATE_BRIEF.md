# Candidate Brief

Refine the incumbent: `erlang_interval_size_family_prior` — rank 0, the best model on this data, ELPD-LOO -2260.7. Its hypothesis, as its author stated it:

> Refinement of `exception_rule_family_prior_2`: people do the same Bayesian concept learning over rules and intervals, with the size principle and a separate prior mass per concept family (exception rules in their own family), but they do not regard all numeric intervals as equally plausible a priori. The one change is an Erlang-shaped prior over interval width with a learned characteristic size (as in Tenenbaum's number game): very narrow and very wide ranges are a priori unlikely, so after a single example or tightly clustered examples people generalise to a band of moderate width around them rather than collapsing onto the tightest interval, and after widely spread examples broad ranges are penalised relative to natural rules.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/erlang_interval_size_family_prior.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
