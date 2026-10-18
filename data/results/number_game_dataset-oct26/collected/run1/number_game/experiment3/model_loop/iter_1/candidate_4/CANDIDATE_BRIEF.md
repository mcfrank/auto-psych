# Candidate Brief

Refine the incumbent: `shortfall_from_mutual_similarity` — rank 0, the best model on this data, ELPD-LOO -4725.5. Its hypothesis, as its author stated it:

> A refinement of `mutual_similarity_contrast_standard`: people still hold the target to a standard set by how much the examples resemble one another on the final-digit metric (same digit most alike, same parity partly, opposite parity least). The one change is in how the target is compared with that standard: what counts against a "yes" is the shortfall, how far the target's own similarity to the examples falls below their similarity to one another, and nothing else. A target that resembles the examples at least as much as they resemble each other pays no penalty, partial matches pay only for the gap that is left, and in the incumbent's model a target with no digit match pays the full mutual similarity regardless of its parity credit, which no longer happens. Rule-pre-empted parity credit, the common-factor boost and rule-gated decade-band familiarity are kept unchanged.

Its source is `/scr/wurgaft/auto-psych/agent_trees/78ecf8bc0448807b/repo/_runs/cell_1/experiment3/model_loop/models/shortfall_from_mutual_similarity.py`. Read it before you write anything.

Produce the version of this model you believe would beat it on the current data: keep the mechanism that makes it win and change what it gets wrong — a different functional form, prior or normalisation of its mechanism, a cue it ignores, or a component taken from another model in `refinement_menu.md`. For this slot the rule against grafting cues from other models onto a hypothesis is lifted, and so is the rule against composing mechanisms: you may add a component from another model. Two things still hold: make **one** deliberate, stated change (a grab-bag of cues added to fit better is not a refinement), and change something that matters — the novelty gate rejects a candidate whose predictions match a live model's across the stimulus space, the incumbent's included.

Say in `hypothesis.md` which model you refined and what you changed: that is part of the claim.

If `critiques.md` is present, prioritise a hypothesis that addresses one of the significant discrepancies it reports.
