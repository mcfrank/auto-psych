---
name: session-heterogeneity
description: exp2 responses.csv = 10 sessions x 450 (trial_index resets); sessions 1-5 yes~0.42, 6-10 yes~0.70; models have no individual differences
metadata:
  type: project
---
Exp2 data has no participant_id, but trial_index resets 0-449 split it into 10 sessions in row order (the PPC harness keeps row order). Sessions 1-5 say yes ~0.42 (only 53-80% yes to shown examples); sessions 6-10 ~0.70 (~85-94% to examples). Every incumbent so far is single-population. iter_4 critique (2026-10-19) submitted session_yes_rate_sd and session_example_yes_sd to test this; check iter_4/critique/ppc_results.json.
**Why:** after 32 null stats on stimulus features, between-person heterogeneity is the most plausible remaining misfit. **How to apply:** if the session SDs are significant, candidates should add per-session (or two-group) bias/lapse parameters. Related: [[ppc-critiques-null]], [[decision-rule-mechanisms-null]]
Outcome (iter_4 critique): both session SD stats were null (p=0.60 and p=0.99). The single-population model already reproduces the spread, which is probably driven by the stimuli, so per-session parameters are not motivated.
Outcome (iter_4 ppc): session_yes_rate_sd obs 0.144 vs null 0.147 (p=0.60), session_example_yes_sd p=0.99 — the incumbent reproduces between-session spread (it comes from stimulus differences), so no per-session parameters are needed.
