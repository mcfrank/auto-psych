---
name: output-dispersion-crowding-veto
description: exp3: targets within ~3 of an example endorsed less (multi in-span d<=3 ~.33 vs .47); crowding-veto graft on parity_doubt_three_example_memory
metadata:
  type: project
---
exp3 iter_4 c2 (2026-11-02): raw data show multi-example targets close to an example (dist 1-3) endorsed ~.30-.43 vs ~.47 at dist 5-35 inside span; near(<=3) overall .366 vs .398; reversals of examples .33; same decade .372. n=1 shows no dip. Built output_dispersion_crowding_veto = incumbent × (1 - crowd·exp(-(d-1)/w)); check OK, smoke -4659.2 (~tie with incumbent -4658.5), ~4.5 min per smoke run.
Also: sum/diff-of-examples targets .457 vs .391; multiple-of-an-example .463 vs .382 (untested mechanisms).
**Why:** early_session_yes_gap critique is unmodellable (trial_index banned, [[guessing-lapse-ceiling]]), so the exploratory slot went to a new effect.
**How to apply:** check the fit result before grafting again; arithmetic-closure (sum/diff) is an open idea. Related [[exp3-iter4-critique-stats]].
