# Candidate Brief

Refine the incumbent: `mixed_strong_weak_sampling_concepts` — rank 0, the best model on this data, ELPD-LOO -2283.9. Its hypothesis, as its author stated it:

> Refinement of `beta_share_width_exception_concepts`: people do the same Bayesian concept learning over rules (family-organised prior, exception rules as their own kind) and magnitude ranges judged as a typical share of the 1–100 line, with occasional stray examples. The one change is their assumption about how the examples were generated: they believe only some genuine examples (an inferred share) were drawn at random from the concept, which is what drives the size principle, while the rest were merely numbers the concept happens to contain, chosen without regard to how big the concept is (mixed strong/weak sampling). So the size principle is diluted most for big concepts, and after examples that fit both a narrow and a broad concept (multiples of 9 vs multiples of 3, a tight range vs a wide one) people keep the broad concept in play and say "yes" to its other members more than pure random sampling predicts.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/mixed_strong_weak_sampling_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
