# Candidate Brief

Refine the incumbent: `personal_complement_symmetry_aversion` — rank 0, the best model on this data, ELPD-LOO -3465.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent `complement_symmetry_aversion`: people still find a sequence that equals its own reverse with heads and tails swapped (HHTT, TTTTHHHH, HHTTHHTT, HTHTHTHT) deliberately constructed and so less random, on top of the incumbent's judgement (personal ideal switching rate and sensitivity with lenience toward over-alternation, personal weights on imbalance and the longest streak, shared weights on the final streak, mirror symmetry, run-length variety and the most lopsided stretch, and a personal guessing rate). The one change is that noticing this antisymmetry is a personal trait rather than shared: each person has their own weight on it, drawn from a population, so some people consistently reject antisymmetric sequences while others do not see the pattern at all — predicting that people differ in how often they agree with the majority choice on pairs involving an antisymmetric sequence (the critique's excess spread of majority agreement).

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/models/personal_complement_symmetry_aversion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
