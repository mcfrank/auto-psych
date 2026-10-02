# Candidate Brief

Refine the incumbent: `complement_symmetry_aversion` — rank 0, the best model on this data, ELPD-LOO -3470.5. Its hypothesis, as its author stated it:

> People notice complement symmetry (antisymmetry): a sequence of four or more flips whose second half is the first half read backwards with heads and tails swapped — it equals its own reverse with H and T exchanged, as in HHTT, TTTTHHHH, HHTTHHTT, HHHTHTTT or strict alternation HTHTHTHT — looks deliberately constructed, and so less random, by a shared weight. This sits on top of the current best judgement (each person's own ideal switching rate and sensitivity with lenience toward over-alternation, personal weights on heads/tails imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, and a personal guessing rate), which registers mirror symmetry (palindromes) but not this second kind of symmetry, so it predicts people reject antisymmetric sequences more often than the best model does even when their switching rate, balance and streaks look acceptable.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/complement_symmetry_aversion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
