# Candidate Brief

Refine the incumbent: `person_specific_motif_suspicion` — rank 0, the best model on this data, ELPD-LOO -2105.7. Its hypothesis, as its author stated it:

> Refinement of the incumbent `person_specific_chance_switch_belief`: people still judge a sequence random to the extent their own model of a fair coin (with a person-specific belief about how often it switches) explains it better than the regular generators (a switch-biased Markov coin, a biased coin, and a repeating short motif with occasional slips), with the generators' unknowns averaged out and the evidence weighed per flip by a fitted power of length. The single change is that people also differ in how readily they suspect a repeating pattern: each person gives the repeating-motif generator their own prior weight among the regular explanations, drawn from a population, so for pattern-suspicious people perfect alternation and period-3/4 motifs look strongly non-random while pattern-blind people judge such sequences mainly by their switches and balance.

Its source is `/scratch/users/kushinm/auto-psych/outer_loop_live/run3/data/subjective_randomness/experiment2/model_loop/models/person_specific_motif_suspicion.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
