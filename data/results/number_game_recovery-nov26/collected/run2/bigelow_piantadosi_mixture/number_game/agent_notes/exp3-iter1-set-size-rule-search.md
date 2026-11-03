---
name: exp3-iter1-set-size-rule-search
description: Exp3 iter-1 exploratory — rule-vs-range prior log-odds shifting with number of examples (kappa), smoke -3193.6, submitted
metadata:
  type: project
---
Exp3 iter 1 candidate_2 (2026-10-31): `set_size_triggered_rule_search` = heading_extrapolation_weber_width + kappa*(n_examples-2.5) added to the rule-vs-range prior log-odds (kappa ~ N(0,1)). Built by string-patching the incumbent file; check OK, smoke LL -3193.6 (no MAP screen done). Exp3 set sizes: 47 one-example, 66 two, 50 three, 55 four sets.
**Why:** set size was unused information in the prior; untested mechanism.
**How to apply:** if it ties/dups, set-size-dependent priors are another null — see [[number-game-ceiling]].
