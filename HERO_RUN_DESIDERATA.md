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

4. Must-fix before spending money (from the jank audit)

- The /results and /submit Cloud Functions are unauthenticated (functions/index.js:177 and :94). Anyone with the publicly-served collection_session_id can read all collected data or inject fabricated participant rows. For a small pilot this was accepted scope; for a hero run that's live for longer with more money at stake, it's the top risk. A shared-secret token check on both endpoints is enough.
- prolific_mode: live has no confirmation gate — it recruits and pays straight from a YAML flag, while the smoke-deploy path requires --confirm-production. Add the symmetric gate to the live launch path.
- Eval-pool mismatch is still the default: seed-holdout evaluates on a 500-pair subsample (configs/holdout_recovery.yaml:32-36, code default at holdout_recovery.py:1135) and only a manual reanalyze_holdout_exhaustive.py pass reconciles it. Set exhaustive: true up front so hero results are never on mismatched pools.
- Loud-failure violations: project_id defaults to "" instead of raising (collect.py:677,915, validators.py:43), and ground_truth.py:33 still swallows exceptions. Also worth verifying src/runtime/prolific.py's blanket except Exception wrappers re-raise rather than return falsy defaults. Good news from the audit: the all-left steering bug is genuinely fixed, and degenerate-data collection now raises loudly.
- Smaller: unify the inconsistent MCMC defaults across entry points (outer 2000/2000/4, standalone inner 500/500/2, design-time twin fits hard-coded 500/500/2 at run.py:268, yaml 3000/2000/4); no per-participant consent record is written to Firestore (possible IRB/audit gap).
