---
name: exception-tolerant-three-example-memory
description: exp3 iter_4 c4: n-1-of-n tolerance grafted on parity_doubt_three_example_memory incumbent; check OK, smoke -4656.6 (~+2)
metadata:
  type: project
---
exp3 iter_4 candidate_4 (2026-11-02): `exception_tolerant_three_example_memory`. The incumbent parity_doubt_three_example_memory with one addition: near-shared properties (held by all but one of 3+ examples, the remembered 3-subsets included) enter at a Beta tolerance fraction of their salience. Check OK, smoke loglik -4656.6 against roughly -4658.5 for the incumbent. The check took about 11 min, with two 330 s sampling runs.
The only critique that survived FDR, early_session_yes_gap, needs trial_index, which models may not use. See [[guessing-lapse-ceiling]].
**Why:** the two best tied grafts are now composed. Expect a small gain.
**How to apply:** if this is admitted, the next round should not re-graft tolerance or the WM limit. Related: [[exception-tolerant-parity-doubt]] [[parity-doubt-three-example-memory]]
