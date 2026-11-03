# Candidate Brief

Refine the incumbent: `uncertain_reliability_noise_concepts` — rank 0, the best model on this data, ELPD-LOO -835.7. Its hypothesis, as its author stated it:

> Refinement of `inferred_example_noise_concepts`: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, allowing that some examples are stray draws from 1-100 — but they do not know the program's reliability, so instead of assuming one fixed stray rate they hold a belief over how reliable the program is and average over it for each example set. The one change is this uncertainty about the noise rate, which makes strays exchangeable but correlated within a set: one example that breaks a tight rule makes people suspect the program is unreliable on that set, so they discount the remaining examples too and generalise more broadly, while a set fully consistent with a tight rule is taken as strong evidence for it.

Its source is `/scr/wurgaft/auto-psych/agent_trees/f8fb9bba6e6fa0a9/repo/_runs/cell_1/experiment1/model_loop/models/uncertain_reliability_noise_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
