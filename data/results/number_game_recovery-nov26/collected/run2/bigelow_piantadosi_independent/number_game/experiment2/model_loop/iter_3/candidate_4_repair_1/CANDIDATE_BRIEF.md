# Candidate Brief

Refine the incumbent: `base_ten_multiples_capacity_limited` — rank 0, the best model on this data, ELPD-LOO -2083.6. Its hypothesis, as its author stated it:

> Refinement of `capacity_limited_evidence_integration`: people do the same capacity-limited Bayesian learning over rules and numerical ranges, with each added example contributing less evidence, but "multiples of 5" is not just one more multiples rule sharing a single plausibility with multiples of 3, 4, 7, 11 and the rest — because it is aligned with our base-ten number notation, it is its own, far more readily entertained kind of rule with a separate prior plausibility inferred from the data. The single change is splitting multiples of 5 out of the multiples family (as even and odd are already split out of parity), predicting that sets such as 90, 100, 5 are generalized to every multiple of 5 across the number line, while sets fitting a less familiar multiple (e.g. 42, 35, 21, 84 as multiples of 7) are generalized by that rule less than the incumbent predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment2/model_loop/models/base_ten_multiples_capacity_limited.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
