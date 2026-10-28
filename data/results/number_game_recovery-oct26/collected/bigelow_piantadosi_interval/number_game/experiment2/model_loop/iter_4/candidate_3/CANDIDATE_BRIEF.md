# Candidate Brief

Refine the incumbent: `volatile_concept_changepoint` — rank 0, the best model on this data, ELPD-LOO -1941.0. Its hypothesis, as its author stated it:

> People read the examples one at a time and update their belief about the program's concept after each one, but they believe the program might switch to a different concept between outputs: before each new example they allow, with an inferred switch probability, that the concept has been re-drawn from their prior. This belief in a volatile generator makes recent examples count more than earlier ones in a graded, Bayes-consistent way — a concept supported only by the last few examples keeps substantial belief even when an earlier example contradicts it — while everything else (weighted rules and ranges, size principle with tempered evidence, first-example weighting, rule-size prior, decision gain and criterion) is as in the best model; a switch probability of zero recovers it.

Its source is `/scr/wurgaft/auto-psych/agent_trees/5af1c6040af4ffd9/repo/_runs/cell_1/experiment2/model_loop/models/volatile_concept_changepoint.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
