# Candidate Brief

Refine the incumbent: `parity_bounded_ambiguity_doubt` — rank 0, the best model on this data, ELPD-LOO -4727.2. Its hypothesis, as its author stated it:

> Refinement of `rule_ambiguity_doubt_all_sizes` by one grafted component, taken from `parity_preserving_random_doubt` and applied at every set size. When the examples fit so many candidate rules that people give up on rule-based judgment (whether one example or several are shown), they do not answer completely flatly: they keep the parity that all the examples share as the one default constraint on a loosely patterned program. Their fallback "yes" goes to targets of that parity at a flat rate and much less often to opposite-parity targets. So rule-ambiguous even sets such as 64, 4 or 96, 48 are endorsed broadly for even targets but not for odd ones, while rule-based judgment (the salience-weighted share of the shared profile with the interval similarity) is unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/parity_bounded_ambiguity_doubt.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
