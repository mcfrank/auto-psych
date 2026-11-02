# Candidate Brief

Refine the incumbent: `first_guess_win_stay_concepts` — rank 0, the best model on this data, ELPD-LOO -866.7. Its hypothesis, as its author stated it:

> People do not weigh all examples together at once; they hold a single working guess about the concept (a rule or a numerical range) that they form from the first example by Bayesian concept learning, and as each further example arrives they keep that guess, with a fitted stickiness, as long as the new example fits it, reconsidering the concept from all examples seen so far only when the new example contradicts the guess (or when they do not stick). Because the guess is anchored on what fitted the early examples, a broad concept chosen after the first example survives later examples that would have favoured a narrower one, so the same examples in a different order produce different generalisations (a primacy effect).

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment1/model_loop/models/first_guess_win_stay_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
