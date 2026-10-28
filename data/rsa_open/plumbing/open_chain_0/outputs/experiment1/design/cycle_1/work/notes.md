# Designer Session Notes (Cycle 1)

## Current Status
- Candidate `cand_bet.json` (sha256 `e3b1f9fcf770...`):
  - Response type: `bet` (`allocation` link, `total_points: 10`)
  - Scene space: `[[2,2], [3,2], [3,3], [4,2], [4,3], [4,4]]`, `prior_queries: true`, `synonyms: true` (1361 scenes)
  - Design: `n_participants: 250`, `trials_per_participant: 16`, `distinct_trials: 48`, `catch_trials: 2`, `minutes: 4.4`, `pay_usd: 0.8` ($266.67)
  - Time estimate: 4.4 min vs 4.4 min declared (exact match, within 30% tolerance)
  - `link-check`: PASSED (`converged=True, recovered=True`, report `link-check_e3b1f9fcf770.json`)
  - Full `eig`: PASSED (`3.791 of 3.907 bits (p_correct 0.73), $266.67, 0.01422 bits/$; floor 0.00643: MEETS`, report `eig_e3b1f9fcf770.json`)
  - `submit`: SUBMITTED `e3b1f9fcf770`.

## Backup Candidates in `work/`
- `cand_choice_rating.json` (two-measure choice + 5-point rating, quick EIG 3.778 bits)
- `cand_choice_5obj.json` (choice up to 5 objects, quick EIG 3.717 bits)
- `cand_production_judgment.json` (speaker production + binary judgment, quick EIG 3.626 bits)

