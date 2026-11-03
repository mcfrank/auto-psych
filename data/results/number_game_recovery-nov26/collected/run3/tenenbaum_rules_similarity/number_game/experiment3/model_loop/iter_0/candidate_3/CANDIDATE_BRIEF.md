# Candidate Brief

Refine the incumbent: `uncertainty_shifted_criterion` — rank 0, the best model on this data, ELPD-LOO -3663.6. Its hypothesis, as its author stated it:

> People infer the hidden concept exactly as the best model's Bayesian concept learner does, but their yes/no decision criterion depends on how sure they are which concept the program uses: the more their belief is spread over many competing concepts (rules and range widths), the more they shift their answer away from the inferred membership probability toward "no" (uncertainty aversion) or, if the inferred shift is negative, toward "yes". So two targets with the same membership probability get different answers depending on whether the examples point clearly to one concept or leave many open.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment3/model_loop/models/uncertainty_shifted_criterion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
