# Candidate Brief

Refine the incumbent: `asymmetric_tolerance_lopsided_stretch` — rank 0, the best model on this data, ELPD-LOO -3500.3. Its hypothesis, as its author stated it:

> Refinement of the incumbent `most_lopsided_stretch_aversion`, grafting in the asymmetric ideal-rate tolerance of `run_variety_asymmetric_ideal_rate`: people still find a sequence less random when one stretch of it shows a striking local heads/tails excess, and still judge it by their own ideal switching rate and sensitivity, personal weights on overall imbalance and the longest streak, shared weights on the final streak, mirror symmetry and run-length variety, and a personal guessing rate. The one change is that a departure from one's ideal switching rate counts differently in the two directions: switching more often than the ideal costs only a shared fraction (fitted, possibly far from one) of what the same shortfall in switching (streakiness) costs, so people are lenient toward over-alternation — predicting that, beyond what the local-excess cue explains, the more alternating of two highly alternating sequences is chosen more often than a symmetric penalty implies.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/asymmetric_tolerance_lopsided_stretch.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
