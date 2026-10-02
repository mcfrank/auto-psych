# Candidate Brief

Refine a model of your choosing — any model in `refinement_menu.md`, which lists every model in this project other than the incumbent `complement_symmetry_aversion` (the incumbent has its own refinement slots this round). The menu gives the other live models, with their standing against the best, and the models pruned earlier, with the margin by which each lost and its source (in the `pruned/` directory of the experiment that pruned it). A pruned model lost on the data it was scored on, but its mechanism may be partly right, and this slot exists to find out. Choose the model whose mechanism you judge most promising and most improvable — a narrow loser over a distant one, unless you see exactly what the distant one got wrong — read its source, and produce the version of it that could overtake the incumbent.

For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, and a pruned model re-implemented as it was would pass the gate only to lose again by the same margin.

Say in `hypothesis.md` which model you chose and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
