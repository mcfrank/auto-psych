# Candidate Brief

Refine the incumbent: `erlang_interval_simplicity_search` — rank 0, the best model on this data, ELPD-LOO -890.4. Its hypothesis, as its author stated it:

> Refinement of `simplicity_cost_concept_search`: people still build concepts from mental primitives, paying a fitted cost per extra primitive for rules, weighing examples by the size principle and committing nearly all-or-none to the best-explaining concept. The one change (taken from `tenenbaum_rules_similarity`) is the prior over numerical intervals: instead of treating every stretch of the number line as equally natural, people favour intervals of a typical width (an Erlang prior on interval size with a fitted scale), so tight ranges hugging the examples are a priori unlikely and rules win more often — predicting less "same decade / nearby number" generalisation and stricter respect for shared properties such as parity than the incumbent.

Its source is `/scr/wurgaft/auto-psych/agent_trees/d6b6238b22d937a5/repo/_runs/cell_1/experiment1/model_loop/models/erlang_interval_simplicity_search.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
