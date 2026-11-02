# Candidate Brief

Refine the incumbent: `beta_share_width_exception_concepts` — rank 0, the best model on this data, ELPD-LOO -2285.2. Its hypothesis, as its author stated it:

> Refinement of `gamma_width_exception_family_concepts`: people do the same Bayesian concept learning over rules (family-organised rule prior, exception rules as their own kind) and magnitude ranges, with the size principle. The one change is the shape of their prior about how wide a range is: they think of a range as a portion of the bounded 1–100 line and expect it to cover a typical share of it (a beta prior over that share, with its mean and sharpness inferred), rather than holding an unbounded gamma expectation over raw width. So ranges covering nearly the whole line are as implausible to them as very narrow ones, and when the examples are spread widely (e.g. 43, 4, 91, 9) people fall back on a rule instead of an "almost any number" range. As a result they say "yes" less often to numbers that only such a near-total range would include.

Its source is `/scr/wurgaft/auto-psych/agent_trees/570a40651ab1ca31/repo/_runs/cell_1/experiment2/model_loop/models/beta_share_width_exception_concepts.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
