# Candidate Brief

Refine the incumbent: `individual_range_sampling_looseness` — rank 0, the best model on this data, ELPD-LOO -3512.2. Its hypothesis, as its author stated it:

> Refinement of `loose_range_sampling_sparse_rules`: people do the same capacity-limited Bayesian learning over arithmetic rules (kind plausibilities, sparse familiarities within kinds) and numerical ranges, reading the examples with one sampling strength for rules and a separate, looser one for stretches of the number line. The single change is that how loosely the examples are read as evidence for a tight range is a stable trait of each participant rather than a population constant (each person's range sampling strength is scaled by a personal factor with a fixed spread around the population, inferred from their answers; the rule strength stays shared). Because the examples' range evidence grows with every example, this predicts individual differences that grow with set size: some participants keep to the examples' span and reject far rule members, others favour the fitting rule, consistently across every set each person sees, while for single examples people barely differ.

Its source is `/scr/wurgaft/auto-psych/agent_trees/fc3d1e4e3afb1c37/repo/_runs/cell_1/experiment3/model_loop/models/individual_range_sampling_looseness.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
