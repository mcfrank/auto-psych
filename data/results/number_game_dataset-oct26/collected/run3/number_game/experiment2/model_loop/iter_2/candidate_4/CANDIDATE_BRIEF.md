# Candidate Brief

Refine the incumbent: `evidence_scarce_guessing_concepts` — rank 0, the best model on this data, ELPD-LOO -2806.0. Its hypothesis, as its author stated it:

> Refinement of `decade_aligned_range_concepts`: people do the same conservative Bayesian concept learning over the 42 arithmetic rules and midpoint-weighted, decade-favoured intervals of 1-100, with evidence weight growing with set size, but when the examples give them little to go on they often do not consult a concept at all and simply guess (with their own inferred leaning toward "yes"), and this guessing becomes rarer as more examples are shown (an inferred guessing rate for a single example and an inferred decline with set size). The one change is this evidence-dependent guessing, which predicts that after a single example people say "yes" fairly often even to targets no plausible concept contains (e.g. 94 or 100 after 7, 1 after 100), whereas after three or four examples such targets are almost always rejected — the excess single-example non-member "yes" rate the critique of the best model flags.

Its source is `/scr/wurgaft/auto-psych/agent_trees/80aefb383357367b/repo/_runs/cell_1/experiment2/model_loop/models/evidence_scarce_guessing_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
