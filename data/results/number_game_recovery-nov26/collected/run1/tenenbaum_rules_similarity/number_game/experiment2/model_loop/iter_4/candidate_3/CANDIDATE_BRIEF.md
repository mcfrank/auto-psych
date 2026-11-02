# Candidate Brief

Refine the incumbent: `round_number_range_boundaries` — rank 0, the best model on this data, ELPD-LOO -2256.7. Its hypothesis, as its author stated it:

> People infer the hidden program as an ideal Bayesian concept learner would (number rules, bounded ranges and one-sided thresholds, weighed by family plausibility, range width and the size principle, with examples drawn without repetition), but their sense of how plausible a bounded range is is biased by round numbers: multiples of ten act as cognitive reference points, so a range whose lower or upper boundary falls exactly on one (e.g. 20-37 or 40-60) is judged more, or less, likely a priori than an otherwise identical range, with the direction and strength of this round-number bias inferred from the data. This shifts where people stop saying "yes" near multiples of ten — for example whether 40 is included after examples in the low thirties — while leaving rule-based generalisation unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/round_number_range_boundaries.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
