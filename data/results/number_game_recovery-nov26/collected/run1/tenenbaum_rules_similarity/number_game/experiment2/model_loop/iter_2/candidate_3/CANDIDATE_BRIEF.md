# Candidate Brief

Refine the incumbent: `threshold_concepts_erlang_interval` — rank 0, the best model on this data, ELPD-LOO -2258.9. Its hypothesis, as its author stated it:

> Refinement of `erlang_interval_size_family_prior`: people do the same Bayesian concept learning over rules and numeric intervals, with the size principle, a prior mass per concept family (exception rules in their own family) and an Erlang prior over the width of bounded ranges, but they treat one-sided magnitude thresholds ("numbers above 60", "numbers below 20") as a distinct kind of program rather than as just very wide intervals that happen to touch 1 or 100. The one change is that these threshold concepts form their own family with a separately inferred prior mass, shared equally among them and exempt from the width prior, so when all the examples are large (or all small) people extend "yes" all the way to the end of the number line, not just to a band of moderate width around the examples.

Its source is `/scr/wurgaft/auto-psych/agent_trees/043c9ecd8d452a7b/repo/_runs/cell_1/experiment2/model_loop/models/threshold_concepts_erlang_interval.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
