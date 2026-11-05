# Candidate Brief

Refine the incumbent: `parity_doubt_three_example_memory` — rank 0, the best model on this data, ELPD-LOO -4721.8. Its hypothesis, as its author stated it:

> Refinement of the incumbent `parity_bounded_ambiguity_doubt` by one grafted component, taken from `working_memory_three_example_limit`: people can hold only three examples in mind, so when four are shown, on an inferred share of trials one example has dropped out of memory and the whole judgment — the salience-weighted share of the shared profile, how many rules the examples fit (and so how often people give up and answer flatly), and the parity kept in that fallback — is made from the three remembered examples, averaged over which one was lost. Because three examples share more coincidental properties and fit more rules than four, four-example sets are judged against a richer profile, so targets fitting the full set's profile are endorsed less, and sometimes more broadly, than a reasoner using all four would; sets of one to three examples are judged exactly as by the incumbent.

Its source is `/scr/wurgaft/auto-psych/agent_trees/2485af553ddb9dd3/repo/_runs/cell_1/experiment3/model_loop/models/parity_doubt_three_example_memory.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
