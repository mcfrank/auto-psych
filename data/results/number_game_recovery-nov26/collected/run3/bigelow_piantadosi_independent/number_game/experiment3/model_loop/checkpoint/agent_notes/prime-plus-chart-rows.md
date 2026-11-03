---
name: prime-plus-chart-rows
description: exp2 iter_2 cand_5 prime_salient + hundred-chart row bonus; smoke loglik -2141.8, ~2 min per 200-draw chain
metadata:
  type: project
---
2026-10-30, cell_1/experiment2 iter_2/candidate_5 (`prime_salient_chart_rows`): the hundred_chart_row_ranges incumbent plus prime_bonus (rule index 2). This is aimed at the FDR-surviving critique prime_set_prime_target_lift (observed 0.22 vs null -0.19). The smoke fit gave loglik -2141.8 and the check passed. Built by string-patching the incumbent source.

**Why:** the prime bonus helped on its own ([[prime-salience-finding]]), and the incumbent lacks it.
**How to apply:** check whether this candidate was admitted before adding the prime bonus to the incumbent again.
