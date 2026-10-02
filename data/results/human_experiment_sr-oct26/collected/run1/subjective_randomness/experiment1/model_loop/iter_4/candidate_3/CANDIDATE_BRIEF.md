# Candidate Brief

Refine the incumbent: `lapse_individual_balance_alternation` — rank 0, the best model on this data, ELPD-LOO -952.4. Its hypothesis, as its author stated it:

> Refinement of `individual_balance_ideal_alternation` (statistically tied with the incumbent): each person still judges a sequence as more random the closer its proportion of alternations lies to their own personal ideal switching rate, and lopsided heads/tails counts make a sequence look less random by a weight that is each person's own. The one change is grafting in the incumbent's decision rule: on some trials a person does not compare the sequences at all and guesses, at a personal trait-like lapse rate, so indifferent participants are explained by guessing rather than by weak preferences, and the balance and alternation preferences of engaged people can be as sharp as the data show — addressing the critique that among alternation-matched pairs people pick the sequence without a long streak (usually the more balanced one) more often than the lapse-only incumbent predicts.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment1/model_loop/models/lapse_individual_balance_alternation.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
