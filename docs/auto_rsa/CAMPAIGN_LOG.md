# auto-rsa campaign log: decisions and runs

The one place to start from when writing up the campaign. It has three parts:
- every design decision with its reason and how it compares with main (§2–3);
- every step and run in order (§4);
- what is still open (§5).

Details live in the linked docs; this file only summarises and points. `PLAN.md` holds the
same decisions with their full text, by date.

**Keeping it current:**
- Every PI decision and every run or step gets a line here, in the same commit as the change.
- Dates are 2026; "PI" means Michael Frank's call.
- In the "vs main" column:
  - **same** means main does the same;
  - **differs** says what main does;
  - **RSA only** means main has nothing that corresponds.

## 1. What the campaign is for

Extend auto-psych from subjective randomness to rational-speech-act (RSA) models of reference
games, and support two claims:

1. **Claim 1 (existing data):** inner-loop runs on the published data yield models that predict
   held-out conditions better than the starting models do. Supported by Sherlock run 2 (§4).
2. **Claim 2 (live data):** the model the loop **commits to before new data** predicts new
   people better than the bar: the five starting models and the 12 promoted seeds. It is scored
   prospectively on each new experiment (PI 10-10).
   - This is a stricter criterion than main's series and the earlier paper drafts, which
     reported in-sample or cross-validated fit.

## 2. Decisions

### 2.1 Platform, models, agents

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 1 | 10-06 | Own branch `auto-rsa`, Python 3.12; models in **memo**, fitted with numpyro NUTS | memo needs Python ≥ 3.12 and compiles to JAX, so it is differentiable; PyMC cannot import memo | **differs:** main is Python 3.11 + PyMC | `PLAN.md` 10-06 |
| 2 | 10-06 | A parallel RSA loop (`src/rsa/loop/`) reusing main's domain-neutral parts, not a backend seam inside main's files | keeps main's runs and tests untouched | **differs** (architecture) | `PLAN.md` Phases |
| 3 | 10-06 | Model contract: agents write only `choice_probs`; the harness owns the likelihood; every model needs a lapse or noise term | pure RSA gives probability 0 to objects people do choose | RSA only | `PLAN.md` Architecture |
| 4 | 10-07 | Agents: Gemini 3.8 Flash via opencode, 2,400 s per agent, sandboxed (bubblewrap) | credits; smoke tests | same sandbox; different model | `SMOKE_RESULTS_2.md` |
| 5 | 10-08 | **Agents have no network** | in run 1 the agents read the pragmods GitHub repository (70 API calls), OSF and papers | **differs:** main's agents can browse (their URLs are reported) | `SHERLOCK_RUN1_RESULTS.md` |
| 6 | 10-06 | Agents never see held-out data, provenance or a simulated run's ground truth | validity | same | `HANDOFF_rehearsal2.md` |
| 7 | 10-06 → 10-10 | **The critique step (main's CriticAL) runs before every inner-loop round in the live phase:** 8 agent-written test statistics scored against 1,000 posterior-predictive replicates of the incumbent; the significant discrepancies go into every candidate's brief. Deferred on 10-06, so runs 1–2 and rehearsals 1–2 ran without it | one framework across phenomena: the paper plays up the critique loop (PI) | **same** as main from the live phase on (main's code and settings; RSA's own data frame and replicates) | `src/rsa/loop/critique.py`, `PLAN.md` "Critique step" |
| 8 | 10-06 | Slot roles (exploratory lenses, refine the incumbent, refine a model of the agent's choice), the ledger of tried hypotheses, starting models not protected | reused from main | same | `src/rsa/loop/` |

### 2.2 Data

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 9 | 10-07 | Existing data: adults' forced-choice listener data from five sources: pragmods, Sikos 2021, Mayn & Demberg 2022/2023/2026 | the comparable paradigm | RSA only | `DATASETS.md` |
| 10 | 10-07 | Excluded: child data, imagined-child and LLM speakers, sliders, feedback studies (Duff 2026), Franke & Degen 2016, Franke et al. 2024 | PI: not the same task or population | RSA only | `DATASETS.md` |
| 11 | 10-07 | Unlicensed and CC BY-NC-ND trial data are derived at run time, never committed; aggregates may be | licences | RSA only | `DATASETS.md` |
| 12 | 10-07 | Held-out set: conditions held out *within* papers (~20% of each source) | claim 1's test; whole-paper holdout would leave sources unseen | RSA only (main tests on simulated recovery) | `src/rsa/split.py` |

### 2.3 The inner loop on existing data (claim 1)

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 13 | 10-07 | Prune unit: a **display within an experimental condition**, at 2 clustered SEs | by display alone, pragmods' shared display made one cluster of thousands of trials, and nothing was ever pruned | **differs:** main clusters by stimulus (2 SEs in recovery sweeps, 4 in its October live series) | `PLAN.md` "Pruning unit" |
| 14 | 10-08 | **Selection by grouped 5-fold CV** over held-out units: ranking, incumbent, prune and export (admission keeps the PSIS-LOO gates) | in run 1, trial-level PSIS-LOO pruned all six of the best held-out models | **differs:** main ranks by ELPD-LOO among reliable fits | `PLAN.md` 10-08 |
| 15 | 10-08 | Live-set cap 12 | in run 1 the cap of 8, not the prune, decided what survived | **differs:** main 8 | `PLAN.md` 10-08 |
| 16 | 10-08 | Recovery verdict: the exported model within pool RMSE 0.01 of the ground truth | the held-out clause had no power | RSA only | `PLAN.md` 10-08 |
| 17 | 10-08 | Per-source standings are descriptive only (reports and agents' briefs), never in a selection rule | avoid tuning selection to a source | RSA only | `PLAN.md` 10-08 |
| 18 | 10-08 | **Stop after 2 rounds without a new best** (`max_iterations` stays the ceiling) | on run 2's history, stopping after 1 would have cut real gains; 2 changes nothing in runs 1–2 | **differs:** main always runs every round | `PLAN.md` 10-08 |
| 19 | 10-07 | Fit aggregated (display, choice) patterns, not trials | smoke test 2 ran out of memory; 40k trials are 501 patterns, and a fit went from 853 s to 50–75 s | RSA only (exact same likelihood) | `SMOKE_RESULTS_2.md` |

### 2.4 Seeds and chains for the live phase

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 20 | 10-08 | **Promote seeds from run 2:** all 143 admitted models refitted on all data, clustered into 10 groups, the best of each by grouped CV, plus `rsa_l2` | the design can only aim at disagreements between the models it has; a pre-specified step, not a theory choice | **differs:** main seeds from a fixed literature registry | `PLAN.md` 10-08; `src/rsa/promote.py` |
| 21 | 10-09 | **Add `rsa_l1` as a seed** (12 seeds), instead of running a recursion-depth test ourselves | whether depth matters is for the loop to find; 3.5 nats behind `rsa_l2` held out | RSA only | `PLAN.md` 10-08 |
| 22 | 10-09 | **Split the seeds across 3 chains:** `rsa_l2` and `rsa_l1` in every chain, the other 10 dealt 4/3/3 so that the closest pair within any chain is as far apart as possible | identical seeds would give three copies of one chain; the split separates the three plain-display near-twins. The chains are diversified starts, not replicates | **differs:** main's runs share one seed set | `src/rsa/outer/chains.py` |
| 23 | 10-09 | Claim-2 bar in every chain: all 12 promoted seeds + the 5 starting models | one bar for all chains | RSA only | `OuterConfig.promoted` |

### 2.5 The live design

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 24 | 10-06 | **Many trials per person:** 10 designed displays + 2 catch + 1 practice, each person a balanced subset of the design | pragmods is 1 critical trial per person; 5–10× more data per dollar | **differs:** in main everyone answers every stimulus | `PLAN.md` 10-06, 10-08 |
| 25 | 10-09 | **200 people, 40 displays, 50 responses per display** per experiment | power is flat in N and D (0.76–0.79); more displays help downstream: CV folds, clustered SEs, agents' conditions | **differs:** main uses 64 stimuli × N | `PLAN.md` 10-08 |
| 26 | 10-08 | **Plain displays only** (no valence, familiarization or greyscale); the existing trials on such displays (1,313, about 2.6%) are left out of live-phase fits | valence/familiarization models cannot be told apart on plain displays; the live displays vary none of it | RSA only (scope) | `PLAN.md` 10-10 |
| 27 | 10-10 | **The design aims at the carried models and the bar**: half the prior on each, duplicates merged, a simulated ground truth withheld unnamed | in rehearsal 1 the displays tested only the carried models, never the bar that claim 2 is scored against | **differs:** main designs over the carried set under a uniform prior | `LOOP_ALGORITHM.md` §2.2 |
| 28 | 10-10 | **Quotas, with EIG choosing inside them:** of 40, at least 6 two-object, 10 three-object, 8 mumble, 20 with a word; the free design is recorded beside it | free EIG put 31–38 of 40 on 4×4 (and 37 of 40 on mumble trials in one experiment); cognitive load likely differs by size (PI). Measured cost: power 0.760 vs 0.767 free (±0.007) | **differs:** main's EIG is unconstrained | `PLAN.md` "Design mixture" |
| 29 | 10-10 | **Wider display pool:** features shared by exactly the same objects, plus 2×3 and 2×4 displays (794 → 1,597 displays) | RSA's speaker counts every word, so a redundant feature shifts its predictions by about 10 points while heuristics don't move; more two-object displays | RSA only | `PLAN.md` "Design mixture" |
| 30 | 10-10 | No feature that no object has; the heard word is always true of some object | invisible to participants, and undefined for the literal listener | RSA only | `PLAN.md` "Design mixture" |
| 31 | 10-10 | Agents get **no say in the design**; they are told what the displays are (sizes, extrapolation risk, mumble trials) and that the speaker's alternatives are theirs to propose (saying nothing, a word for a feature nobody has) | keeps the optimal-design half of the paradigm; educate, don't mandate (PI) | same (main's agents don't design either) | `brief.PLAIN_SCOPE_NOTE` |
| 32 | 10-10 | Same-hypothesis threshold: 0.002 RMSE on the pool (novelty, carry-forward, design merges), rechecked on the wider pool | calibrated: twins 0.0002, `rsa_l1` vs `rsa_l2` 0.0070–0.0086 | same number as main, on RSA's pool and choice-class probabilities | `src/rsa/design/distinct.py` |

### 2.6 The live loop and claim 2

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 33 | 10-08 | The loop fits the existing data plus every live experiment so far | models stay answerable to the literature | **differs:** main has no literature data | `PLAN.md` 10-08 |
| 34 | 10-10 | **Guarded selection:** eligible only within 4 clustered SEs of the best on the existing data's CV; eligible models ranked on the live trials; ineligible ones pruned | in rehearsal 1, cumulative selection never moved off the literature, and live-only selection collapsed onto near-copies that predicted worse than plain RSA | **differs** | `PLAN.md` 10-10 |
| 35 | 10-10 | Selection is cumulative over the live experiments (L1 ∪ … ∪ Ln) | PI | RSA only | `LOOP_ALGORITHM.md` |
| 36 | 10-10 | **Carry forward up to 8 distinct models,** best first; a model within 0.002 RMSE of one kept is dropped | rehearsal 1 carried 12 near-copies and its next design had power 0.25 | **differs:** main carries every survivor (cap 8), with no distinctness filter | `carry.json` |
| 37 | 10-10 | The tried-hypotheses list carries across experiments | the agents don't re-propose | same | `LoopConfig.inherit_ledger` |
| 38 | 10-10 | Only a new candidate's admission fit is time-limited (30 min); models in play fit to completion | a promoted seed's fit passing 30 min killed rehearsal 1 | same | `fitting.NO_TIME_LIMIT` |
| 39 | 10-10 | **Claim 2's test:** the committed model (the previous export) vs the best starting model, the best promoted seed and each bar model, by lpd on the new experiment, with SEs clustered by designed display | success = predicting new people prospectively | **differs:** main tracks incumbent changes and in-sample or cross-validated fit | `prospective.json` |
| 40 | 10-09 | Participants who miss any catch trial are excluded | data quality | RSA only | `max_catch_errors: 0` |
| 41 | 10-10 | Recovery in simulated runs: mean KL per display from the ground truth (pool and designed displays), with RMSE beside it; the ground truth must be coherent with the human data | rehearsal 1's L0 ground truth was rejected by the data, and pool RMSE improved while prediction got worse | RSA only | `recovery.json` |

### 2.7 Participants, money, privacy, cluster

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 42 | 10-07 | IRB: the subjective-randomness protocol covers this; the repo's consent text, shown by the deployment's consent gate | | same | `HANDOFF_live.md` |
| 43 | 10-09 | $12/h for an estimated 4 min ($0.80, about $1.06 with Prolific's fee); about **$1,915** for 3 chains × 3 experiments × 200; a pilot of about 20 people (about $21) measures the real time | | same rate | `rsa_live.yaml` |
| 44 | 10-10 | **Exclude only earlier RSA participants** | subjective randomness is a different enough task (PI) | **differs:** main excludes every earlier auto-psych study | `exclude_earlier_participants_from: project` |
| 45 | 10-10 | Prolific ids only in each cell's `private/`, never copied off Sherlock; trial data stay on Sherlock until de-identified sharing is decided | privacy | same principle | `HANDOFF_live.md` |
| 46 | 10-10 | Live recruitment double-gated (config + flag) plus a typed "yes"; stages: test deploy → pilot → campaign, each with PI go-ahead | real money | same | `HANDOFF_live.md` |
| 47 | 10-10 | Server-assigned trial lists (`/assign`, round robin in arrival order); `/results?format=json`; the page posts trials as one JSON string | balanced lists; Firestore refuses nested arrays (stage 1) | RSA only (additive functions) | `functions/`, `src/rsa/live/` |
| 48 | 10-09 | **One core per process** (fits, agents, the harness) and `OPENBLAS_NUM_THREADS=1` | Research Computing flagged 140–160 threads per process | same aim as main's BLAS pinning; RSA also covers JAX | `src/rsa/cpus.py`, `HANDOFF_threads.md` |
| 49 | 10-09 | **Dense mass matrix** in NUTS for the live loop, and JAX's persistent compile cache | same posteriors, 2.5–9× fewer sampling steps (494 s → 149 s on the slowest model); repeated compiles read back | RSA only | `PLAN.md` "Compute" |
| 50 | 10-10 | Agent spend recorded per inner loop and per run | rehearsal 1 recorded none | same | `token_usage_summary.json` |

## 3. How this differs from main, at a glance

- **Stack:** Python 3.12 + memo/JAX/numpyro instead of 3.11 + PyMC, in a parallel loop.
- **Agents:** no network. The critique step is main's from the live phase on (claim 1's runs had none).
- **Data:**
  - literature data are always in the fit;
  - claim 1 is scored on conditions held out within papers;
  - live data are many trials per person, with balanced subsets.
- **Selection:**
  - grouped CV instead of ELPD-LOO;
  - guarded live selection;
  - stop after 2 stale rounds;
  - live cap 12;
  - prune by display within condition.
- **Seeds:** promoted from our own run 2 and split across three chains; main uses one literature seed set.
- **Design:**
  - over the carried models and the bar, half the prior each;
  - quotas with EIG inside them;
  - a wider RSA-specific pool;
  - 200 × 10 of 40 displays.
- **Carry-forward:** at most 8 *distinct* models; main carries every survivor.
- **Claim 2:** the committed model's prospective test against the bar; main tracks incumbent
  changes and in-sample fit.
- **Recruitment:** blocks earlier RSA participants only.

## 4. Steps and runs, in order

| When | Step | Where | What | Headline | Led to | Doc / data |
|---|---|---|---|---|---|---|
| 10-06 | Core, data, seed comparison | cloud | memo core; pragmods as trial rows (6,703); five starting models compared | salience-L1 best; L2 beats L1 by ~13 nats; E6 valence unexplained | the loop | `data/rsa/seed_comparison_all` |
| 10-06/07 | Smoke test 1 | cloud | 1 round × 3 slots, pragmods, Gemini 3.1 Pro, unsandboxed | 2/3 admitted; `rsa_l2_salience` +17.7 nats; self-check never ran; $0.71 | self-check fixes, agent timeout, a new prune unit | `SMOKE_RESULTS.md` |
| 10-07 | Datasets | cloud | four more sources added, 61k rows | licences sorted | #9–11 | `DATASETS.md` |
| 10-07 | Smoke test 2 | cloud | 2 rounds × 3, all sources, Gemini 3.8 Flash, sandboxed | 6/6 admitted, +119 nats; repair works; OOM in round 2; $4.72 | aggregated fitting (#19), `--resume` | `SMOKE_RESULTS_2.md` |
| 10-07 | **Sherlock run 1** | Sherlock, array 46902403 | 6 cells: real ×2, recovery (literal, salience) ×2; 5 × 6 | real: −7.5 (SE 71) and +83.2 (SE 27) held out vs the best seed; all 4 recovery cells recovered; agents browsed; $313 | grouped CV, no network, cap 12, RMSE verdict (#5, 14–17) | `SHERLOCK_RUN1_RESULTS.md` |
| 10-07/08 | **Sherlock run 2 (claim 1)** | Sherlock, array 46968325 | 7 cells: real ×3 (8 rounds), recovery ×2 ×2 (4 rounds); grouped CV; no network | **held out vs `rsa_l2`: +84.3 (SE 36), +56.2 (SE 24), +26.4 (SE 13)**; gains on pragmods and Sikos; all 4 recovery cells recovered; $452 | stale-round stop, 36G per cell (#18) | `SHERLOCK_RUN2_RESULTS.md` |
| 10-08 | Existing-data phase report | cloud | claim 1 per cell and per source, E6 sensitivity, recovery | without E6: +41.4 / +47.9 / +34.5 | | `data/rsa/existing_data_report.html` |
| 10-08 | Run 2 archived | Sherlock 47042589 | 5.1 GB to Oak; key scan clean | | | `HANDOFF_sherlock_promote.md` |
| 10-08 | **Seed promotion** | Sherlock 47042590 | 143 models → 10 groups → best by CV, + `rsa_l2` | 11 seeds (12 with `rsa_l1`, 10-09) | #20–21 | `data/rsa/live_seeds/` |
| 10-08 | Design power | Sherlock 47043328 | D = 10/20/30 × N = 100/200/300 | power 0.76–0.79, flat; capped by three near-twin seeds | #22, #25 | `data/rsa/live_seeds/design_t10/` |
| 10-08/09 | Thread incident | Sherlock | Research Computing flagged job 47042590 | 140–160 threads per process | one core per process (#48) | `HANDOFF_threads.md` |
| 10-09 | Compute tuning | cloud | dense mass, compile cache | slowest fit 494 → 149 s | #49 | `PLAN.md` "Compute" |
| 10-09 | **Chains** | Sherlock 47075879 | seeds split 6/5/5 | power at N=200, D=40: 0.982 / 0.849 / 0.978 | the chain seeds | `data/rsa/live_seeds/chains/` |
| 10-08/09 | **Rehearsal 1** | Sherlock, array 47081376 | chain 0, 3 experiments × 200 simulated people; cumulative vs live-only selection; ground truth `literal_listener` | cumulative never moved; live-only collapsed onto near-copies; experiment 3's design had power 0.25; crashed on a 30-min fit limit; spend not recorded | guarded selection, bar-aware design, distinct carry, no limit on models in play, KL recovery, spend, plain scope, committed-model claim 2 (#26–27, 34–41) | `REHEARSAL_REVIEW.md` |
| 10-10 | Live wiring | cloud | Firebase `/assign` and `/results` JSON, page, collection, launcher, config | e2e tests | | `HANDOFF_live.md` |
| 10-09/10 | **Rehearsal 2** | Sherlock 47176690, 15 h 20 min, completed | chain 0, guarded, 2 experiments, coherent ground truth | design power 0.889 → 0.840; **claim 2 +147.0 (SE 43) vs best starting, +49.3 (SE 15) vs best promoted seed**; per-display KL to the truth 0.0122 → 0.00021 (exp 1), 0.00017 (exp 2); exp 2 stopped after 2 stale rounds; $83 for 60 agent calls; not evidence about recovery (the ground truth is the agents' own family) | KL label bug found and fixed (86906717); carry-cap question (§5) | `REHEARSAL2_REPORT.md`, `data/rsa/rehearsal2/` |
| 10-10 | **Live stage 1** (test deploy) | Sherlock 47242197 | chain 0's design deployed; Prolific draft | page live, draft made; **not passed:** data didn't save (Firestore nested arrays); 38 of 40 displays were 4×4; a `.secrets~` backup in staging (never read) | JSON-string trials, secrets exclusions, RSA-only blocklist, quotas and wider pool (#28–31, 44, 47) | `STAGE1_FINDINGS.md` |
| 10-10 | Quota comparison + model check | cloud | chain 0's experiment-1 design, free vs quotas; all 505 model files on the wider pool | power 0.760 vs 0.767 (±0.007); 6 vs 1 two-object displays; every model defined | quotas adopted as defaults | `PLAN.md` "Design mixture" |
| 10-10 | Critique step built | cloud | main's CriticAL ported (#7); e2e tests with scripted agents | 28k trials: 12 s for 1,000 replicates, ~12 s per statistic | rehearsal 3 | `PLAN.md` "Critique step" |
| next | **Rehearsal 3** (short) | Sherlock | chain 0, 1 experiment, 2 rounds, rehearsal 2's ground truth: the critique and the new design with real agents | | | `HANDOFF_rehearsal3.md` |
| 10-10 | **Live stage 1 re-run: passed** | Sherlock 47272640, 49 min | chain 0's design with quotas, deployed; Prolific draft `6acaaa90…` | quotas met (6 two-object, 10 three-object, 13 mumble, 27 word); power 0.932 vs 0.931 free; page works end to end (consent → redirect; word, mumble and two-object displays; a list per id, the same on reload); data saved as one JSON string | the pilot can go (PI) | `STAGE1_FINDINGS.md` §9 |
| next | Pilot (about 20 people, about $21) | Sherlock + Prolific | PI go-ahead | | | `HANDOFF_live.md` §3 |
| next | Campaign (3 chains × 3 experiments × 200, about $1,915) | Sherlock + Prolific | PI go-ahead after the pilot | | | `HANDOFF_live.md` §4 |

## 5. Open

- **Rehearsal 3:** decided yes, short (PI 10-10), to check the critique step with real agents
  (`HANDOFF_rehearsal3.md`). The campaign waits for its report.
- **Carry cap and families:** rehearsal 2 kept six close relatives and dropped two distinct
  models over the cap. One option is to fill the cap across families first. No change made.
- **A recovery test with an unfamiliar, pre-fixed ground truth:** optional, for the paper. It is
  simulated, so it can run alongside the live campaign.
- **The real completion time:** the pilot measures it, and it sets the pay. The welcome screen
  says "about 3 minutes", Prolific's estimate is 4, and the PI's run took 1–2.
- **The consent overlay** (main's deployment gate) is not in the page's accessibility tree:
  its "I agree" button is pointer-only (stage 1 re-run note).
- **Known limits to report:**
  - simulated people are less noisy than real ones;
  - pruning on the live rows is weak;
  - valence and familiarization terms sit at their priors on plain data.
