# Subjective-Randomness Recovery

This directory contains the project-level machinery for checking whether the
loop can re-identify the model that generated simulated choices.

The importable library code lives in `src/subjective_randomness/`; these are the
runnable command-line entry points. The PyMC model families and the manifest
that says which of them are active live in
`src/subjective_randomness/pymc_model_families/`, their pure-Python twins in
`src/subjective_randomness/model_families/`. The examples below write their
results under `data/subjective_randomness/`.

## Holdout Recovery — the Full Agentic Loop vs. a Held-Out Ground Truth

*Holdout recovery* keeps the true model *out of* the candidate set: each
ground-truth model in turn generates every synthetic response from fixed
parameters, while the full agentic outer+inner loop starts from the live seed
pool and tries to recover the held-out process.

Ground truths come from the recovery registry
(`src/subjective_randomness/pymc_model_families`, the config's
`seed_models_dir`), whose pure-Python family twins provide the fixed generating
parameters. The **live seed pool** mirrors that registry's
manifest, so a ground truth drawn from the active set is genuinely held out of
experiment 1's seed pool. The seed baselines load each seed from the files
the cell was seeded with (experiment 1's zoo, or its `pruned/` once the
loop pruned a seed there), not from the registry, so a run is re-scored with
the seeds it ran with. Since 2026-09-28 the loop prunes starting models like
any other model; `holdout.json` records `starting_models_prunable` (absent in
cells run before, which never pruned them), and results of the two
conditions must not be pooled. A ground truth outside the registry manifest (an
impossible theory) is absent from the pool already, and nothing is excluded.

The real pipeline does the work: the programmatic exhaustive design chooses
each experiment's stimuli by joint EIG, the model set is carried forward
between experiments (there is no theory agent — new hypotheses enter only via
the inner loop), and the inner loop's candidate agents conjecture new PyMC
models that are fit by MCMC and compared by ELPD-LOO. What crosses an
experiment boundary is the loop's **live set** — the seeds plus every model
still within the pruning margin of the best, not only the winner — together
with the ledger of every hypothesis tried (`attempted_hypotheses.jsonl`), and
the next design's model prior is uniform over that carried set.

After **every inner-loop scoring step** (the initial seed-set fit and each
candidate round, in every experiment) the then-best model's posterior-predictive
`p_left` is correlated with the ground truth's `p_left` on a large held-out
stimulus set, giving a trajectory of how the loop converges on the true process.

```bash
uv run python scripts/subjective_randomness/holdout_recovery.py \
  --config scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml \
  --out data/subjective_randomness/holdout_recovery/holdout.json \
  --tidy-csv data/subjective_randomness/holdout_recovery/holdout.csv \
  --figure data/subjective_randomness/holdout_recovery/holdout.png
```

This spawns real coding agents and runs real MCMC — a full 3-model run takes
hours. Scope a smoke run first:

```bash
# 1 ground truth, 1 experiment, seed-set-only inner loop, tiny MCMC
uv run python scripts/subjective_randomness/holdout_recovery.py \
  --config scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml \
  --out /tmp/holdout_smoke/holdout.json \
  --gt-model motif_stack \
  --n-experiments 1 --n-participants 5 --inner-loop-iterations 0 \
  --draws 150 --tune 150 --chains 2
```

Outputs, per held-out model under `<out dir>/<out stem>_runs/<gt_model>/`:
the full `experiment1..N/` pipeline trees, `eval_stimuli.json` (the held-out
evaluation set), and `trajectory.json` (the per-step correlation trajectory,
per-experiment model sets, and a leakage audit). A held-out pair on which a
scored model's `p_left` is not a probability (NaN, or outside [0, 1]) is left
out of that step's metrics (and a fitted seed's undefined pairs out of that
seed's baseline metrics): the row records `n_eval_excluded` and
`eval_excluded_models` (columns of the tidy CSV too), the log prints a
`[eval] WARNING` line saying the metrics cover fewer pairs than the baselines,
and every excluded pair is listed in `eval_exclusions.jsonl` beside
`trajectory.json`. The sweep summary (`holdout_test_retest.py`) lists every
cell and step where this happened.
The combined JSON, tidy CSV
(one row per `gt_model × step`: `gt_model, experiment, step, iteration,
global_step, best_model, pearson_r, rmse`), and correlation-vs-step figure land
at the paths you pass.

### Raw-only pipeline

The pipeline carries only raw H/T sequences — there is no featurizer. Every
model (seeds and candidates alike) must compute the features it uses via a
`compute_features(sequence_a, sequence_b)` hook. This ensures recovery measures
whether a model can *discover* its decision variable, not just regress on a
harness-supplied column that reproduces the ground truth at R² 0.90–1.00.

Details worth knowing:

- **Held-out eval set.** The EIG design picks training stimuli from anywhere
  in the pair space, so
  holdout is enforced *after* the run: any eval-pool pair that appeared (in
  either order) in any of the run's training data is dropped, and the run fails
  loudly if fewer than `eval_pool.min_remaining` stimuli survive. The pool is
  **exhaustive by default** (every distinct same-length pair for
  `eval_pool.lengths`, with `predict_max_draws` thinning the posterior so the
  prediction array stays bounded) so seed-holdout and impossible-holdout runs
  are always evaluated on comparable pools; set `eval_pool.exhaustive: false`
  plus `n_pairs` for a sampled pool.
- **Novelty gate.** A candidate is rejected at admission when its posterior-
  mean `p_left` is within `inner_loop.novelty_rmse_threshold` (default 0.002)
  RMSE of an admitted model's, measured on the loop's own novelty pool (512
  same-length pairs at lengths 4–8, generated from a fixed loop seed and
  recorded as `model_loop/novelty_pool.json`) — never on the training stimuli
  and never on the eval pool above. Set the key in the config, pass
  `--novelty-rmse-threshold` to the script, or export `NOVELTY_RMSE_THRESHOLD`
  to the sbatch array to A/B it; `0` disables the gate. The value used is
  recorded under `inner_loop` in the result JSON.
- **Inner-loop scale.** `inner_loop.max_iterations` (candidate rounds per
  experiment) and `inner_loop.candidate_count` (slots per round) are 5 and 6
  in both holdout configs since September 2026 (30 proposals per experiment;
  the earlier sweeps ran 2 x 3). `inner_loop.n_critique_proposals` (default
  8) is how many test statistics the critique agent proposes per round. All
  three can be overridden per run (`--inner-loop-iterations`,
  `--inner-loop-candidates`, `--n-critique-proposals`; sbatch env
  `INNER_LOOP_ITERATIONS`, `INNER_LOOP_CANDIDATES`, `N_CRITIQUE_PROPOSALS`)
  and every value used is recorded under `inner_loop` in the result JSON.
  A round's candidate agents run concurrently (one worker per slot); the
  critique agent runs before them because its critique is inlined into the
  candidate briefs.
- **Per-step history.** The inner loop now writes `model_loop/history.json`
  (best model + posterior after the seed fit and after every candidate round;
  `best_model` is selected exactly as the export is — ELPD-LOO rank among
  PSIS-LOO-reliable models — with the raw `argmax_model` and the
  `excluded_unreliable` list recorded beside it);
  the trajectory evaluation refits each step's best model through the shared
  MCMC cache (`--cache-dir`, default `<out dir>/mcmc_cache`), so evaluation
  costs no new sampling.
- **The cache ignores fit kwargs.** Cached fits are keyed by model file + data
  only, so changing `--draws`/`--tune`/`--chains` for a fresh run requires
  clearing the cache directory first.
- **The held-out model's identity never enters the agents' tree.** The
  synthetic `data/responses.csv` (and the pooled `model_loop/responses.csv`
  derived from it) is written without the generator's `generating_model`
  column — that column is listed in every candidate's and critic's context, so
  it used to tell the agents which model to rediscover (`strip_generating_model`;
  the harness refuses a responses file that still carries it). The Slurm array
  task additionally deletes the held-out model's `.py` from both model
  directories the agents can open and removes its manifest entry (name plus
  mechanism rationale) with `remove_manifest_entry.py`.
- **Residual leakage is audited, not prevented.** `trajectory.json` flags
  byte-identical copies of the ground truth's source, mentions of its
  distinctive parameter names, and files named after it — heuristics for
  auditing a run, not proof it was clean.
- **Design = exhaustive joint EIG.** Each experiment's design is the same
  programmatic exhaustive selection as the live pipeline: every H/T pair over
  the design lengths is scored under the experiment's actual PyMC model set
  and the jointly most informative set is picked greedily (lazy batched
  greedy in float32 since 2026-09-27: ~3 min instead of 11-13 h for a later
  experiment; `validate_lazy_eig.py` compares it with exact greedy on a
  finished design). No agent is
  involved; the design is deterministic given the models and registry.
- **Resume after a failure.** The harness stops loudly at the first
  stage whose output doesn't validate. Re-run the same command with `--resume`
  to continue: any ground truth that already has a `trajectory.json` is
  skipped, and within an incomplete run every stage whose output already
  validates is skipped, so work restarts at the first invalid stage.
  An unfinished model-loop stage (no `model_loop/export_complete.json`, the
  record written after the export and the registry) is redone from scratch:
  `model_loop/` is wiped (it is fully regenerable, and its MCMC fits are still
  in `mcmc_cache/`), and `cognitive_models/` and the run's `agent_notes/` are
  put back to what the stage first started from (`cognitive_models_input/`,
  `agent_notes_at_start/`), so notes about the abandoned attempt's candidates
  are discarded.
  If `--resume` finds a `trajectory.json` whose experiment count disagrees with
  the config's `n_experiments`, it fails loudly rather than mixing runs.

  ```bash
  # after fixing the cause of the failure, continue the same run:
  uv run python scripts/subjective_randomness/holdout_recovery.py \
    --config scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml \
    --out data/subjective_randomness/holdout_recovery/holdout.json \
    --tidy-csv data/subjective_randomness/holdout_recovery/holdout.csv \
    --figure data/subjective_randomness/holdout_recovery/holdout.png \
    --resume
  ```

- **opencode permissions.** Agents run from the repo root, so opencode treats
  `/tmp` (and `/private/tmp`, `/var/folders`) as external directories and, in
  non-interactive `opencode run`, auto-rejects writes there — which silently
  breaks agents that stage temp files. The repo `opencode.json` grants
  `external_directory` allow rules for those paths so this does not recur. If
  you point the pipeline at a different scratch directory, add it there too.

## Model Families

The registry holds the four literature-faithful models of the 2026-08
consolidation. Each has a PyMC adapter in
`src/subjective_randomness/pymc_model_families/` and a pure-Python twin in
`src/subjective_randomness/model_families/`, checked against paper-derived
test vectors in `tests/test_literature_model_families.py`; the manifest's
`rationale` entries state each mechanism.

| Model | Faithful to |
| --- | --- |
| `falk_konold_dp` | Falk & Konold (1997) Difficulty Predictor, minimal parse |
| `motif_stack` | Griffiths et al. (2018) four-motif stack automaton |
| `finite_experience_occurrence` | Hahn & Warren (2009) occurrence probability in finite experience |
| `local_representativeness` | Explicit quantitative operationalization of Kahneman & Tversky (1972) |

All four share one forced-choice observation model. For a trial with left
sequence `A` and right sequence `B`, each model computes a sequence-level
score `S(seq; theta)` and predicts:

```text
P(choose left | A, B, theta) =
  sigmoid(beta * (S(A; theta) - S(B; theta)) + side_bias)
```

`beta` is choice sensitivity (higher means more deterministic choices) and
`side_bias` a left/right response bias (positive values favor the left
sequence, independent of its content). `motif_stack` and
`finite_experience_occurrence` are defined only for same-length pairs.
