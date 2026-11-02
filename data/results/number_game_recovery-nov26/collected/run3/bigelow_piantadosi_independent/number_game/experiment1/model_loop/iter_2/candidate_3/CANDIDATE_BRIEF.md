# Candidate Brief

Refine the incumbent: `simplicity_cost_concept_search` — rank 0, the best model on this data, ELPD-LOO -899.2. Its hypothesis, as its author stated it:

> People search for a concept by building it from simple mental primitives, and every extra primitive a concept needs costs them: the simplest concepts (odd, even) come to mind readily, while compound concepts (multiples of 7, ending in 2, powers of 3, a rule with an exception) are penalised in proportion to how many primitives describe them, on top of the size principle and the near-all-or-none commitment to the best-explaining concept. This complexity cost makes people fall back on broad parity rules much more than the size principle alone predicts — e.g. saying yes to any even number after 22, 62, 26, 82 rather than only to numbers ending in 2.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/simplicity_cost_concept_search.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
