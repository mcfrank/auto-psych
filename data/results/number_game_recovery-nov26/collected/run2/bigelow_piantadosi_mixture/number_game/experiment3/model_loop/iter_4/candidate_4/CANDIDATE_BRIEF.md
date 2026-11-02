# Candidate Brief

Refine the incumbent: `position_decaying_example_evidence` — rank 0, the best model on this data, ELPD-LOO -3195.0. Its hypothesis, as its author stated it:

> People read the examples one after another and build a running guess about the program, but each example revises that guess by a steadily changing amount: the evidence an example carries shrinks (or grows) by a constant factor with each step down the sequence, with the factor inferred, so the total weight of the evidence is not fixed but depends on how many examples came before. Otherwise they do plain Bayesian concept learning over arithmetic, digit and digit-sum rules and every numeric range of 1-100 (equally plausible ranges, independent draws with some strays, occasional guessing). This predicts that a later example narrows generalization less than a full-weight learner expects when evidence fades (for example, a fourth example after 31, 3, 1), and that the same numbers shown in a different order generalize toward the concepts favoured by whichever examples came first.

Its source is `/scr/wurgaft/auto-psych/agent_trees/88c1bd3cf1e5f7a8/repo/_runs/cell_1/experiment3/model_loop/models/position_decaying_example_evidence.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
