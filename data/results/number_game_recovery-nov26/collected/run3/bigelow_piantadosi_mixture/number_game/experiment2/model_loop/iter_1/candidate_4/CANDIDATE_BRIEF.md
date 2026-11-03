# Candidate Brief

Refine the incumbent: `first_example_anchored_reliability` — rank 0, the best model on this data, ELPD-LOO -2054.1. Its hypothesis, as its author stated it:

> People do Bayesian concept learning over mathematical rules and magnitude ranges of 1-100 (size principle, a prior over rules set by kind of rule, an uncertain belief about how often an example is a stray draw from 1-100), but they treat the first example differently from the rest: the first number shown anchors the concept and is trusted to come from it at its own, inferred reliability, while only the later examples are pooled into the uncertain judgement of how reliable the program is on that set. So a stray-looking number shown first constrains the concept far more (or less) than the same number shown later, and the same set in a different order generalises differently.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment2/model_loop/models/first_example_anchored_reliability.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
