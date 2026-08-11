# Things to fix

- Fix the git merge issue in the implementation prompt
- Get rid of the design candidates only prompt (and option)
- Remove the outer loop theorist agent
- Computing EIG with respect to a uniform prior over models is wrong, but the actual posterior is overconfident. We should do something more principled.
- Inner loop models that get exported should get descriptive file names.

# Things to improve

- Maybe give more context to the inner loop theorist directly in the prompt rather than leaving files in its context (e.g. inject the critiques and instructions into its prompt directly)
- Figure out how to explore more broadly
- There should probably be some gating to make sure a model is genuinely novel before adding it to the registry.
- Maybe add some kind of pruning of models that obviously lose to avoid re-fitting the same bad models over and over again.
- Scale up the number of inner loop candidates and iterations

# Jank to clean up (from Claude code review)

Must-fix-before-spending-money items from the jank audit that are **still open**:

- Loud-failure violation: `project_id` defaults to `""` instead of raising
  (`src/pipelines/outer_loop/collect.py:640`, `src/validation/validators.py:43`).
- The MCMC defaults are inconsistent across entry points (outer 2000/2000/4,
  standalone inner 500/500/2, design-time twin fits hard-coded 500/500/2, yaml
  3000/2000/4). Unify them.
- No per-participant consent record is written to Firestore (possible IRB/audit gap).

The rest of that audit is resolved (verified 2026-08; see git history): the
`/results` and `/register_session` Cloud Functions now require the
`x-results-token` shared secret; `prolific_mode: live` requires
`--confirm-live-recruitment`; all three `scripts/subjective_randomness/configs/holdout_recovery*.yaml`
set `exhaustive: true`, so seed-holdout no longer evaluates on a mismatched
500-pair subsample; `ground_truth.py` raises on a broken project asset instead of
swallowing the exception, and `src/runtime/prolific.py` documents one error
contract (API failures return an error string; config and programming errors
raise). The all-left steering bug and degenerate-data detection were already
fixed at audit time.
