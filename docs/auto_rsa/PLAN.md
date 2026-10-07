# auto-rsa: auto-psych for rational-speech-act reference games

Branch `auto-rsa`. Goal: generalize the auto-psych discovery loop from
subjective randomness to RSA models of pragmatic reasoning in reference games,
with models written in [memo](https://github.com/kach/memo) and the pragmods
experiments (Frank, Emilsson, Peloquin, Goodman & Potts,
[langcog/pragmods](https://github.com/langcog/pragmods)) as seed data and
paradigm.

This file is the running plan and the record of decisions; update it as phases
land. Handoffs to local sessions (Sherlock, live runs) should point here.

## Decisions (2026-10-06)

| Decision | Choice | Why |
|---|---|---|
| Branch | `auto-rsa`, Python 3.12 | memo requires >= 3.12. `arviz<1`, `pymc<6` caps keep the validated 0.x stack; the `rsa` group's markers keep main resolvable on 3.11 |
| Model language | memo, fitted with numpyro NUTS | memo compiles to JAX, so it is differentiable; PyMC would need a JAX-op bridge and cannot import memo on 3.11 |
| JAX on Sherlock | jax/jaxlib 0.7.0, numpyro 0.19.0 | the last jaxlib with glibc 2.17 wheels; memo 1.3 verified on it. Fallback: an Apptainer image |
| Design | multi-trial within participant | pragmods is 1 critical trial per person (~50 per cell, CI +/-0.11); fresh contexts per trial give ~5-10x more data per dollar |
| Agents | Gemini via opencode (the default) | credits; bump to newer Gemini models as they ship |
| Cluster | a local session manages Sherlock and live runs, from handoffs | cloud sessions cannot reach Sherlock (Duo 2FA), Prolific or Firebase |

## Decisions (2026-10-07)

| Decision | Choice |
|---|---|
| Agents | Gemini 3.8 Flash (`google/gemini-3.8-flash`), 2,400 s per agent; opencode snapshots off; 15 min shell timeout |
| Pruning unit | display within experimental condition, 2 clustered SEs (see "Pruning unit") |
| Held-out evaluation | hold out conditions *within* papers (~20% of each source's trials; a unit is a condition in one-shot experiments and an item in multi-trial ones): `src/rsa/split.py`, `src/rsa/evaluate_heldout.py` |
| Recovery tests | ground truths `literal_listener` and `rsa_l1_salience` (with its near-twin `rsa_l1_shared_prior` also withheld), simulated on the real displays: `src/rsa/simulate.py`, `src/rsa/recovery.py` |
| First Sherlock sweep | 3 conditions (real, recovery_literal, recovery_salience) x 2 replicates, 5 rounds x 6 slots (`HANDOFF_sherlock_run1.md`) |
| Datasets | adults only; forced-choice listener data; no imagined-child/LLM speakers, sliders, feedback studies, or Franke & Degen 2016; unlicensed and CC BY-NC-ND sets are derived at run time, never committed; aggregates from them (report bundles, per-condition tables) may be committed (2026-10-07) |
| Experiment stimuli | the pragmods artwork; identical objects keep pragmods' different base tints (people see slightly different twins, models treat them as identical: an accepted, unnameable difference) |
| IRB | the subjective-randomness protocol covers the RSA pilot; reuse the repo's consent text |

## Architecture

- **Plug-in seams, not a fork.** The outer and inner loop machinery (artifact
  passing, ledger, registry, sandbox, agent launcher, pruning, LOO
  reliability, convergence gate, fit cache) is domain-neutral. About 20 files
  hard-code binary choice (`chose_left`/`p_left`), PyMC or H/T stimuli. They
  move behind a `Domain` interface (response columns, stimulus pools,
  validators, prompts, jsPsych template, cluster key) and a `ModelBackend`
  interface (load, contract, fit -> InferenceData, predict choice probabilities,
  simulate). Subjective randomness stays the first implementation and keeps its
  tests green.
- **`src/rsa/`** is the domain's standalone library (as `src/subjective_randomness/`):
  - `context.py`: contexts, per-shape grouping, the sink utterance.
  - `model_file.py`: the memo model contract.
  - `fit.py`: numpyro fit producing InferenceData.
  - `pragmods_ingest.py`: the seed data.
- **Project assets:** `src/pipelines/outer_loop/projects/rsa_reference/`
  holds `seed_models/` (memo), `data/pragmods_trials.csv` (canonical
  trial-level seed data) and, to come, `problem_definition.md`,
  `task_description.md`, `references/`, and the experiment template.

### The model contract (agents write only the cognitive part)

A model file defines `PARAMS` (numpyro priors) and `choice_probs(params, ctx)`,
which returns the probability of choosing each object for one trial. The
harness owns the likelihood (Categorical), vmapping, shape grouping and the
domains `OBJ`/`UTT`, which it injects per context shape. A model never sees
padding. The contract check needs no sampling. At prior draws, the
probabilities must have shape (N_OBJ,), be finite and non-negative, sum to 1,
and have finite gradients. Before sampling, every observed choice must have
non-zero probability (a model needs a lapse or noise process; pure RSA gives
probability 0 to objects the word is false of, and people do choose them).

### memo gotchas found so far (for the agent primer)

- **Constants:** a Python constant inside memo must be escaped: `log(p + {EPS})`.
  A bare `EPS` reads as a choice.
- **Array parameters:** they are declared `lex: ...` and read through a jitted
  indexer (`at(lex, u, r)`, `vec(prior, r)` in `memo_kit`), never by `lex[u, r]`.
- **`log(0)`:** it gives NaN gradients; use `log(p + {EPS})`.
- **All-zero rows:** a choice whose weights are all zero for some value returns
  zeros forward (memo uses `nan_to_num`) but NaN gradients. That is why
  contexts are not padded and the sink utterance exists.
- **Recursion depth:** it must be a concrete Python int, not a traced value.
- **`@memo(cache=True)`:** it breaks autodiff.
- **Observing:** `observes [x.u] is <literal>` fails; use `observes_that [x.u == 2]`.
- **Source files:** memo reads source with `inspect`; models must be files,
  which the loader registers in `sys.modules`.
- **float32:** probabilities sum to 1 only within ~1e-6.

## Phases

| Phase | Work | Where | Status |
|---|---|---|---|
| 0 | `src/rsa/` core (contexts, contract, numpyro fit) | cloud | **done** |
| 0 | Canonical trial-level pragmods data (`pragmods_trials.csv`) reproducing the paper's counts | cloud | **done** (two models.csv cells are mislabelled in pragmods itself) |
| 0 | Seed comparison on all 6,703 trials, by LOO plus r and RMSE over cells (`data/rsa/seed_comparison_all`) | cloud | **done**. salience-L1 best; L2 beats L1 by ~13 nats; ~12% lapse; E6 valence unexplained |
| 0 | Report page: standing, RMSE by experiment, small multiples, loop timeline (`src/rsa/report.py`) | cloud | **done** |
| 2 | Parallel RSA inner loop `src/rsa/loop/` (user decision 2026-10-06: a parallel loop reusing the domain-neutral parts, not a backend seam in main's files): code gate, cached time-limited fits, admission gates, novelty pool, briefs and memo primer, self-check, orchestrator, CLI | cloud | **done**, tested with a scripted fake agent |
| 2 | Smoke test with real Gemini agents (`HANDOFF_smoke_test.md`) | cloud | **done** (`SMOKE_RESULTS.md`): 2 of 3 slots admitted first time, `rsa_l2_salience` the new best (+17.7 nats); the self-check never ran inside an agent (CLI bug, opencode's 120 s shell timeout; both fixed); repair path not yet exercised |
| 2 | Re-run the smoke test with the fixes (2 rounds; see `SMOKE_RESULTS.md` recommendations) | cloud | next |
| 2 | Production inner loop on Sherlock (sandboxed agents) | local session | after the smoke test |
| 2 | Critique step for memo models (PPC test statistics) | cloud | later |
| 3 | Outer loop, simulated: design from the context pool, K-way EIG (Monte Carlo over outcomes), multi-trial designs, recovery against held-out RSA variants | Sherlock | |
| 4 | Live: jsPsych port of the pragmods display, Firestore schema, Prolific pilot | local session | |
| - | More seed data (`DATASETS.md`): Franke & Degen 2016, Mayn & Demberg, Sikos et al. 2021 need OSF/PLoS access | PI / local | |

## Pruning unit (decision 2026-10-07)

The end-of-run prune drops a model more than 2 clustered SEs (ELPD-LOO) behind the best trusted model. A cluster is a display within an experimental condition (`CLUSTER_COLUMNS` in `src/rsa/loop/orchestrator.py`): by display alone, the pragmods simple display shared by many experiments made one cluster of thousands of trials and nothing was ever pruned (literal listener 554 nats behind, SE 324). By condition it is 4.4 SEs behind; close rivals whose advantage is concentrated in a few conditions (salience vs vanilla RSA, 1.2 SEs) are kept.

## Design space (for phase 3)

Distinct object x feature matrices (up to row/column permutation, every word
true of some object, distinct meanings):

| Size | Distinct matrices | With a depth-1 implicature |
|---|---|---|
| 3x3 | 10 | 6 |
| 4x3 | 39 | 28 |
| 4x4 | 97 | 81 (191 word/target pairs) |

At 3 objects the paper covered nearly everything. The first pool is matrices
up to 4x4 x word x prior manipulation (familiarization base rate). Later:
salience, cost, multi-word utterances.

## Environment notes (cloud sessions)

- **Reachable:** public GitHub via git, PyPI, the Gemini and Anthropic APIs.
- **Blocked:** arxiv, OSF, the Stanford paper hosts, ACM. Add them under the
  environment's Network access, or commit PDFs to `projects/rsa_reference/references/`.
- **opencode agents:** the shell tool kills a command after 120 s unless the
  call passes `timeout` (ms); each agent's `.xdg_data/opencode/snapshot` is a
  git snapshot of its working tree (~1 GB per agent from the repo root).
- **Not installed:** `rsync` and `bwrap`. Agent sandboxing, and the tests that
  stage agent trees, need them, so real loop runs go to Sherlock.
- **Secrets:** environment variables set in the environment settings reach
  *new* sessions only.
