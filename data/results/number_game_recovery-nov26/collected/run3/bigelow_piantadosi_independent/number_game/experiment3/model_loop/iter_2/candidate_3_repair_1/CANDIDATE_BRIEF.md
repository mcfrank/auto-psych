# Candidate Brief

Refine the incumbent: `set_size_evidence_privileged_fives` — rank 0, the best model on this data, ELPD-LOO -3501.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent `fives_privileged_decoupled_prime_odd`. Everything in it is kept: primitive-built rules with a per-primitive cost, the final-digit discount, costs on powers/cubes and squares, graded times tables, a separate prime bonus and odd cost, the repeated-digit rule, the privileged "multiples of 5" rule, hundred-chart ranges, the size principle and near-all-or-none commitment. The one change is taken from `set_size_graded_evidence_weight` and concerns how people pool evidence. Agreeing examples are more convincing together than one at a time, so the weight each example carries grows with the size of the set at a fitted rate. After a single example people therefore commit only weakly and generalise broadly, and once three or four examples agree they commit to the shared rule or tight range more sharply than an ideal learner would.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment3/model_loop/models/set_size_evidence_privileged_fives.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
