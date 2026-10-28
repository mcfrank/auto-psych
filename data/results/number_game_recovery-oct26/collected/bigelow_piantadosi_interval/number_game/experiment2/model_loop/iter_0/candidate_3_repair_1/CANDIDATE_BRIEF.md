# Candidate Brief

Refine the incumbent: `split_rule_range_tempering` — rank 0, the best model on this data, ELPD-LOO -1971.6. Its hypothesis, as its author stated it:

> Refinement of criterion_shifted_tempered_decision: people do the same Bayesian concept learning over the 42 weighted rules and all intervals of 1-100 (size principle, 10% noisy examples, tempered evidence) and answer through the same decision gain and shifted criterion, but they do not weigh the examples' evidence equally strongly for the two kinds of concept. The one change is that the size principle is tempered separately, by its own inferred weight, for mathematical rules and for numeric ranges: people may treat examples as strong, informative samples of an arithmetic rule (locking onto "powers of 2" after 16, 8, 64) while treating them as only weak evidence about where a magnitude range starts and stops (so range-based generalization stays broad), or vice versa; equal weights recover the incumbent.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/split_rule_range_tempering.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
