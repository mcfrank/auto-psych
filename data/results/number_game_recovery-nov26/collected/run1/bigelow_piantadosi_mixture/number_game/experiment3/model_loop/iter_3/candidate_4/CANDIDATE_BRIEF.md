# Candidate Brief

Refine the incumbent: `gamma_width_distance_hesitant_guess` — rank 0, the best model on this data, ELPD-LOO -3123.0. Its hypothesis, as its author stated it:

> Refinement of the incumbent gamma_width_range_prior_guess (a single working concept guess — a rule or a numerical range with a gamma-shaped prior on its width, shape fitted — formed from the first example by Bayesian concept learning, kept with fitted stickiness, digit rules discounted and salient only once two examples share the digit, a newly noticed coincidence triggering a rethink, fitted trust, and misfits written off more readily the more examples the guess explains). The one change, the component taken from distance_hesitant_concept_extension: people hesitate to extrapolate their guessed concept, so even for a target inside it their willingness to say the program would produce it falls off smoothly (Gaussian-like) with the target's distance on the number line from the nearest example, at a fitted reach. With far members of rules thus discounted at the answer stage, the range-width prior is free to settle where the near-example gradient demands, so the claim is that both a heavy-tailed sense of range width and reluctance to extend far from the examples shape generalisation.

Its source is `/scr/wurgaft/auto-psych/agent_trees/470f328d08510460/repo/_runs/cell_1/experiment3/model_loop/models/gamma_width_distance_hesitant_guess.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
