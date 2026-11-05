# Candidate Brief

Refine the incumbent: `iter2_candidate1` — rank 0, the best model on this data, ELPD-LOO -4136.4. Its hypothesis, as its author stated it:

> People infer the program's concept by the same tempered, sample-size-neglecting Bayesian concept learning as the current best model, with one bias: explaining away. When the examples all share a specific property (all end in 3), a broader property that this one implies (being odd, containing a 3) looks like a by-product of it, not a candidate rule. Each further example that again shows the specific property strengthens that impression, so a rule that strictly contains another rule all the examples fit loses plausibility by an inferred amount per additional example. Targets that have only the broader property (31, 35, 39 after 63, 23, 43) should be endorsed less than the best model predicts, more so for four examples than for two. Targets with the specific property, lone examples, and sets without nested rules should barely change.

Its source is `/scr/wurgaft/auto-psych/agent_trees/8a52ff3d6e8e6ef7/repo/_runs/cell_1/experiment3/model_loop/models/iter2_candidate1.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
