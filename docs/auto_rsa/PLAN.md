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
| 0 | `src/rsa/` core (contexts, contract, numpyro fit) | cloud | **done** (be44848) |
| 0 | Canonical trial-level pragmods data (`pragmods_trials.csv`) reproducing the paper's counts | cloud | in progress |
| 0 | Reproduce the paper's comparison: depth 0/1/2 x prior placement x alpha, by LOO on trial data (the paper used r over 57 cells: r ~ .95 at depth >= 2, alpha-depth tradeoff) | cloud | next |
| 1 | `Domain` + `ModelBackend` seams. `p_left` becomes `p_choice` with K=2 first. Subjective randomness keeps every test green | cloud | |
| 2 | Inner loop on the fixed pragmods data: memo primer + cheat sheet, prompts, import gate for jax/memo, `check_candidate` for memo, novelty gate on choice-probability vectors over a context pool | cloud (small runs with `GOOGLE_API_KEY`); Sherlock for real runs | |
| 3 | Outer loop, simulated: enumerable context pool, K-way EIG (Monte Carlo over outcomes), multi-trial designs, recovery against held-out RSA variants (including "impossible" ones) | Sherlock | |
| 4 | Live: jsPsych port of the pragmods display (stacked PNGs from a JSON spec), Firestore schema, Prolific pilot | local session | |

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
- **Not installed:** `rsync` and `bwrap`. Agent sandboxing, and the tests that
  stage agent trees, need them, so real loop runs go to Sherlock.
- **Secrets:** environment variables set in the environment settings reach
  *new* sessions only.
