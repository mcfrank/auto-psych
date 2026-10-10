# auto-rsa: auto-psych for rational-speech-act reference games

Branch `auto-rsa`. Goal: generalize the auto-psych discovery loop from
subjective randomness to RSA models of pragmatic reasoning in reference games,
with models written in [memo](https://github.com/kach/memo) and the pragmods
experiments (Frank, Emilsson, Peloquin, Goodman & Potts,
[langcog/pragmods](https://github.com/langcog/pragmods)) as seed data and
paradigm.

This file is the running plan and the record of decisions; update it as phases
land. Handoffs to local sessions (Sherlock, live runs) should point here.

**For the write-up, start at `CAMPAIGN_LOG.md`:** every decision with its reason and how it
differs from main, and every run in order, on one page. Add a line there with each new decision or run.

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

## Decisions (2026-10-08, after Sherlock run 1; fixed before run 2)

Run 1 (`SHERLOCK_RUN1_RESULTS.md`, overview `data/rsa/sherlock_run1/overview.html`)
is the pilot that motivated these. They are fixed before run 2's data and
are not to be changed after seeing run 2.

| Decision | Choice |
|---|---|
| Selection | grouped 5-fold cross-validation over training conditions (whole held-out units, as the test set is held out; `src/rsa/loop/cv.py`) for ranking, the incumbent, the end-of-run prune (2 x unit-clustered SE) and the export. Admission keeps the PSIS-LOO gates. No source-specific rule. |
| Live-set cap | 12, ranked by total ELPD-CV (was 8: the cap, not the prune, decided what survived) |
| Run 2 structure | real data x 3 replicates x 8 rounds (run 1's real cells still improved at round 5; replicates differed by ~90 lpd); recovery x 2 replicates x 4 rounds each (flat after round 3); 6 slots per round |
| Per-source results | descriptive only: in reports and in the agents' briefs (where a model leads and lags), never in a selection rule |
| Recovery verdict | the exported model's pool RMSE to the ground truth <= 0.01; the held-out comparison, the closest live model and the seeds' distances are reported, not part of the verdict |
| Agents' network | none: shell commands under a no-internet filter, web tools denied (`src/rsa/loop/no_network.py`) |
| Held-out claim | "better than the starting models on held-out conditions within the same papers" (the split is unchanged: conditions within papers, not whole experiments) |
| Claims the runs are for | (1) runs on existing data yield candidates better than the seeds on held-out conditions; (2) used as seeds for new data collection (outer loop), they make further progress |

## Decisions (2026-10-08, after Sherlock run 2)

| Decision | Choice |
|---|---|
| Stopping rule | a loop ends early once 2 rounds in a row leave its best model unchanged (`LoopConfig.stop_after_stale_rounds`, `STOP_AFTER_STALE_ROUNDS`; 0 = never). Not 1: on run 2's history, 1 would have stopped real_rep1 after round 1 (rsa_l2 still best), real_rep2 after round 5 (+26.7 instead of +56.2) and real_rep3 after round 3; 2 changes nothing in runs 1 or 2. `max_iterations` stays the ceiling. |
| Memory | 36G per cell (run 2's real cells peaked at 22-26 GB of 30); 5 cells at once fit the 192 GB node |

## Decisions (2026-10-08, the live campaign: claim 2)

Fixed before any live data. Report: `data/rsa/existing_data_report.html`.

| Decision | Choice |
|---|---|
| Seeds | `src/rsa/promote.py` (`scripts/rsa/slurm/promote.sbatch`): every model the three run-2 real cells admitted (143: 36 live, 107 pruned; starting models excluded, identical sources once), refitted on all existing data (train + test), grouped by average-linkage clustering of their pool predictions (the novelty gate's RMSE) into **10**, the best of each group by grouped CV on all data, plus `rsa_l2` as the reference and (PI 2026-10-09) `rsa_l1`, standard RSA, only 3.5 nats behind `rsa_l2` on held-out data (SE about 3.4; 0.0 on pragmods): **12 models**. Kept in the mix rather than testing recursion depth ourselves: whether depth matters is for the loop to find. A pre-specified design step, not a theory choice: the EIG design can only aim at disagreements between the models it has. Promoted models are not protected. Sensitivity reported in the paper: the three exported models as the alternative seed set, fitted to the same live data. **Split across the chains** (PI 2026-10-09; `src/rsa/outer/chains.py`, `chains.sbatch`): `rsa_l2` and `rsa_l1` in every chain and the other 10 dealt 4/3/3 (chains of 6/5/5), the split whose closest within-chain pair is farthest apart on the design pool. Identical seed sets would give three copies of one chain (with the existing data dominating, each would keep the same leader); the split puts the three plain-display twins, which are also the three best by CV, in different chains, so each chain starts with a strong model and no untestable tie. The chains are diversified starts, not replicates (runs 1-2 give replicate variability). Each chain's claim-2 bar is all 12 promoted seeds and all five starting models (`OuterConfig.promoted`). |
| Structure | as main's October live series: **3 independent runs x 3 experiments**, an inner loop of **5 rounds x 6 slots** per experiment (main's `max_iterations: 5`, `candidate_count: 6`), with the 2-stale-round stop |
| Participants | **12 test trials each** (PI 2026-10-08: the literature is mostly one-shot; 6 was fine in our experiments, 64 too far): 10 designed displays plus 2 catch trials, and the practice trial. A person sees a balanced subset of the design's D displays (`trial_lists(n_trials=10)`: a shuffled order walked in windows), so each display gets N x 10 / D responses. About 3 minutes at $12/hr: about $0.60 plus Prolific's fee, about $0.80 a person; **200 people per experiment, D = 40 displays, 50 responses a display** (PI 2026-10-09: the power table below is flat in N and D, so the choice rests on what more displays buy downstream: more whole displays per CV fold for live-only selection, more units for the display-clustered SEs of claim 2, more conditions for the agents; and twice the live trials against the existing data's 50k), so 3 x 3 experiments x 200 people is about $1,440 (ceiling $2-4k). Agent costs are covered by credits. A 20-person pilot first. |
| Power | before fixing N and D: designs of 10, 20 and 30 displays for 200 people, each scored at 100, 200 and 300 people on simulated experiments: how often the generating model ends with the highest posterior (no participants; `src/rsa/design/`, `design.sbatch`). Result (2026-10-08, `data/rsa/live_seeds/design_t10/eig.json`): 0.76-0.79 at every D and N, flat in both, because three promoted seeds (`rsa_l2_singleton_distinct_feat_color_l0` and its `_familiar` and `_valence` refinements) differ only in familiarization and valence terms, which are zero on plain displays: each is recovered about a third of the time (mean posterior on the truth 0.34-0.37), and `evaluative_prominence_l2` (valence-modulated) is partly confused with `rsa_l2` (0.6-0.65 and 0.7-0.85). Every other seed is recovered at 0.95-1.0 by N = 100. Only cumulative fitting can separate the three (the existing data's valence and familiarization conditions put them 86-88 nats apart in CV). |
| Claim 2's measure | `experiment<N>/prospective.json` (PI 2026-10-09): each model going into experiment N, fitted only to the data before it, scored on experiment N's new data before anything is refitted. The bar is the best of the **five starting models** fitted to the same data; from experiment 2 on also the best **promoted seed** (beating it is the live loop's own progress). lpd differences, SE clustered by designed display. |
| Data the live loop fits | the existing data plus every live experiment so far (proposed: the models stay answerable to the literature; the alternative is live data only). Open (PI 2026-10-09): live data are a small share of the total (about 2,000 of 52,000 trials after one experiment), so selection on the cumulative data may barely move; the alternative is to fit on everything but select (CV standing, prune, export) on the live trials only (`OuterConfig.selection_scope`, `LoopConfig.selection_source`). **The simulated rehearsal runs both** (`outer_rehearsal.sbatch`, handoff step 4) with a hidden ground truth picked by rule; the PI chooses from its recovery and prospective records. Participants who miss any catch trial are excluded (`max_catch_errors: 0`; PI 2026-10-09: agreed). |
| Manipulations | plain displays only: a heard word or a prior question; no valence, familiarisation or greyscale (PI 2026-10-08: accepted for the first campaign; the valence models cannot be told apart on these displays). |
| Compute (2026-10-09, before the long runs) | measured on 28k existing trials, one CPU per fit: a production fit took 53 s (`rsa_l2`) to 543 s (`softmax_surprisal_isolated_costly_l4`), of which 36-69 s is fixed cost (mostly JAX compilation) and the rest NUTS sampling. (1) **Dense mass matrix** in the live loop (`OuterConfig.dense_mass`, on; `FitSettings.dense_mass`, off by default so run 2's cache keys stand): the same posterior means, higher ESS, 2.5-9x fewer leapfrog steps on the 5-7-parameter promoted models (494 s -> 149 s on the slowest), neutral on 2-parameter `rsa_l2`; the agents' self-check uses it too. Exact NUTS, only better tuned, so no change to what is selected. (2) **JAX's persistent compilation cache**, shared by every RSA job (`_env.sh`): a repeated compile is read back (fixed cost 36 -> 15 s, 57 -> 23 s on a hit); it helps where shapes repeat (pool predictions of carried models, refits, identical first experiments), not across CV folds. Not done: shorter warmup (no gain with dense mass), skipping CV for dominated candidates (changes the selection rule), reusing posteriors across experiments by importance sampling (the folds change with every experiment's data). |
| Bookkeeping | randomised item, words, screen order and bases never reach a model: a converted row is the designed display (every model input equal), its choice the clicked canonical object, and `src.rsa.design.counts.display_counts` tallies the clicks per designed display and class (`tests/test_rsa_experiment_roundtrip.py`). |

## Decisions (2026-10-10, after rehearsal 1)

Rehearsal 1 (`REHEARSAL_REVIEW.md`, outputs in `data/rsa/rehearsal/`) was run with a literal-listener ground truth that the human data reject. The PI cancelled its cumulative arm, and its live-only arm crashed in experiment 3. Decisions:

| Decision | Choice |
|---|---|
| Selection | **guarded** (`OuterConfig.selection_scope`, default; `LoopConfig.selection_guard_dse` = 4): fit on all data; a model is eligible only within 4 clustered SEs of the best on the existing data's grouped CV; eligible models are ranked on the live trials; ineligible ones are pruned at the end of the inner loop. Cumulative never moved off the literature; live-only collapsed onto one family of near-copies whose prospective score on experiment 2 (-112.5 vs the ground truth) was worse than the plain RSA models it had pruned (`rsa_l1` -83.1, `rsa_l2` -91.2). |
| What a design aims at | the carried models **and the bar** (the five starting models and the 12 promoted seeds; `design.run.Args.bar_models_dirs`). Half the prior is on the carried models, and bar duplicates are merged. A simulated run's ground truth is withheld, unnamed. In rehearsal 1 the displays were chosen to break only the carried models, so the prospective score compared them with bar models the displays never tested. |
| Carry-forward | one model per hypothesis the live displays can tell apart (`src/rsa/design/distinct.py`: within 0.002 RMSE on the plain-display pool is the same), best first by the loop's final standing, at most 8 (`OuterConfig.max_carried`; `experiment<N>/carry.json`). Calibrated on human-data fits: rehearsal 1's `twin_*` models are a median 0.0002 apart, while `rsa_l1` vs `rsa_l2` are 0.0070 (recovered at 0.86-0.90). Rehearsal 1's experiment 3 carried 12 near-copies (power 0.25). |
| Fit limits | only a candidate's admission fit is limited (30 min); models in play fit without a limit (`fitting.NO_TIME_LIMIT`). A promoted seed's prospective fit passing 30 min killed rehearsal 1. |
| Recovery measure | mean KL from the ground truth on the design pool and on the experiment's designed displays; pool RMSE kept beside it. The pool RMSE improved while prediction on the designed displays got worse. |
| Agent spend | per inner loop and per run (`token_usage*.json`); rehearsal 1 recorded none. |
| Claim 2's test | the model the loop **committed to before the new data**, the previous inner loop's export, against the bar: the best starting model, the best promoted seed (both picked after the fact, so conservative), and each bar model, by held-out lpd on the new experiment with SEs clustered by designed display (`prospective.json` `committed_vs`). The best of the models going in, picked after the fact, is kept as `live_vs`, a secondary measure. The success criterion is that a model *prospectively* predicts new people better: not the in-sample or cross-validated fits main's series and the earlier paper drafts reported. |
| Scope of the live phase | **plain displays** (`OuterConfig.scope`). The existing trials on valence, familiarization and greyscale displays are left out of every live-phase fit: pragmods E5 base rate, E6 valence, E7 colour and the colour-prior rerun, 1,313 trials locally (about 2.6% of the existing data). Also: novelty is measured on the plain pool, the framing lens is dropped, and the agents are told. To be reported as scope ("the live phase studies the core reference game; those experiments vary what the live displays don't"); claim 1 (run 2) used everything. |
| Live collection (2026-10-10) | built (`src/rsa/live/`, `functions/` `/assign` and `/results?format=json`, `scripts/rsa/live/`, `outer_live.sbatch`; `HANDOFF_live.md`): server-assigned lists in arrival order; the deployment's IRB consent gate; main's deploy, Prolific polling and pause; Prolific ids only in each cell's `private/`. Staged as a test deploy (a draft), a pilot of about 20 people, then the campaign. Cost at $12/h for an estimated 4 minutes: $0.80 reward, about $1.06 with Prolific's fee, so about **$1,915** for 3 chains × 3 experiments × 200 (the earlier $1,440 assumed 3 minutes); the pilot measures the real time. |
| After live stage 1 (PI 2026-10-10) | **Earlier participants:** a live RSA study excludes only earlier *RSA* studies' participants (`exclude_earlier_participants_from: project`); subjective randomness is a different enough task. **Display sizes and mumble trials:** the agents are told what the experiments show and what mumble trials measure (the prior over referents), in `brief.PLAIN_SCOPE_NOTE`. |
| Design mixture (PI 2026-10-10) | **Quotas, EIG inside them** (`OuterConfig.quotas`, `src.rsa.design.eig.Quota`): of 40 displays at least 6 with two objects, 10 with three, 8 mumble trials and 20 with a word; EIG picks within them and the rest freely. Unconstrained EIG put 31 of 40 on 4×4 in rehearsal 1's first design and 37 of 40 on mumble trials in another; the pool was 61% 4×4 contexts and under 1% two-object ones, and the PI expects cognitive load to differ by size. Every design records the free design beside it (`eig.json` `free`: picks, kinds, power), so the quotas' cost is on record. Measured on chain 0's experiment-1 design (15 models after merges, fits to all human trials, local 2026-10-10): power at N=200 **0.760 ± 0.007 with quotas vs 0.767 ± 0.007 free**, joint EIG 3.18 bits both; free put 28 of 40 on 4×4 and 1 on two objects, the quotas 24 and 6. All 505 model files we hold are defined on every display of the wider pool (prior draws). **The live pool widened** (`novelty.plain_pool`, `LIVE_POOL_SIZES`): features with the same extension are allowed (`design_space.games(..., synonyms=True)`; RSA's speaker counts each word, so a redundant feature moves RSA's predictions by about 10 points where the heuristics do not move), and two-object displays with 3 and 4 features: 794 → 1,597 displays, two-object ones 10 → 54. A feature on no object stays out of displays (nobody sees it; a word true of nothing is undefined for the literal listener). **Speaker alternatives** (a null utterance, a word for a feature nobody has) are the agents' to propose; the note says so. Agents get no say in the design itself this campaign. |
| Tried hypotheses across experiments | carried: each inner loop continues the previous experiment's ledger (`LoopConfig.inherit_ledger`). |
| Rehearsal 2 | one guarded run, 2 experiments, chain 0, and a ground truth coherent with the human data: run 2's unpromoted model within 150 nats of the best on grouped CV that is farthest from chain 0's seeds (`ground_truth.py --rule coherent`). Handoff: `HANDOFF_rehearsal2.md`. |

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
| 2 | Re-run the smoke test with the fixes (2 rounds; see `SMOKE_RESULTS.md` recommendations) | cloud | **done** (`SMOKE_RESULTS_2.md`): all five sources, Gemini 3.8 Flash, sandboxed agents; 6/6 admitted, +119 nats over the best seed; repair path works (forced round); prune fires. OOM-killed in round 2 (per-trial fitting of 50k trials that are 268 displays): aggregate to counts before Sherlock |
| 2 | Production inner loop on Sherlock (sandboxed agents) | local session | after the smoke test |
| 2 | Critique step for memo models (PPC test statistics) | cloud | later |
| 3 | Seed promotion from run 2 (`src/rsa/promote.py`) | Sherlock | job written |
| 3 | Design for experiment 1: joint EIG over the 794-display pool with multinomial outcomes (`src/rsa/design/eig.py`, main's estimator: scenarios, leave-one-out, noise-floor stop, single-response fill) and the power table (`src/rsa/design/run.py`, `scripts/rsa/slurm/design.sbatch`) | cloud + Sherlock | built; runs after promotion (`HANDOFF_sherlock_promote.md`) |
| 3 | Outer loop for RSA (`src/rsa/outer/run.py`): per experiment, design from the live set fitted to all data so far, collect (simulated through the page's records and `convert`; live not wired), prospective score, inner loop (5 x 6, stop rule), carry the live set | cloud | **built**, end-to-end test with simulated people and a scripted agent |
| 4 | Live collection in the outer loop: deploy the page (Firebase), Prolific study, collect and convert (main's `deployment/` code), server-side round-robin list assignment | cloud + local | next |
| 4 | Simulated dress rehearsal on Sherlock: one run, 3 experiments, real agents (sandboxed, no network), a hidden ground truth | Sherlock | after the live wiring's dry run |
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

The live phase (2026-10-10) also allows features with the same extension
(`synonyms`) and two-object displays: 2x2 4, 2x3 6, 2x4 9, 3x3 23, 3x4 51,
4x4 230 matrices; 1,597 contexts with the words and prior queries.

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
