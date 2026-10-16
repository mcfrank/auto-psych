# Candidate Brief

Refine the incumbent: `person_specific_chance_switch_belief` — rank 0, the best model on this data, ELPD-LOO -2107.3. Its hypothesis, as its author stated it:

> Refinement of the incumbent `length_normalised_chance_vs_motif`: people still judge a sequence random to the extent their model of a fair coin explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that each person holds their own belief about how often a fair coin switches, drawn from a population distribution, instead of everyone sharing one over-alternating belief — so some people believe chance switches a lot while others believe chance is streaky and choose the sequence with fewer switches, addressing the critique that far more participants prefer fewer-switch sequences than the incumbent's shared belief allows.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_specific_chance_switch_belief.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
