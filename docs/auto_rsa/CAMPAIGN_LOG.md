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
| 7 | 10-06 → 10-11 | **The critique step (main's CriticAL) runs before every inner-loop round in the live phase:** 8 agent-written test statistics scored against 1,000 posterior-predictive replicates of the incumbent; the significant discrepancies go into every candidate's brief. **The critic scores the live trials only** (the rows selection ranks on; PI 10-11). Deferred on 10-06, so runs 1–2 and rehearsals 1–2 ran without it | one framework across phenomena: the paper plays up the critique loop (PI). Live trials only: rehearsal 3's critic, on all 51.5k trials (4.4% live), wrote 3 of 9 statistics about pragmods experiments the live displays can't show, every p sat at the floor, and half of round 2 chased a literature misfit that guarded selection then rejected | **same** as main from the live phase on (main's code and settings; RSA's own data frame and replicates) | `src/rsa/loop/critique.py`, `PLAN.md` "Critique step" |
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
| 24 | 10-06 → 10-11 | **Many trials per person:** 13 designed displays + 2 catch + 1 practice (PI 10-11, after the pilot; was 10 + 2), each person a balanced subset of the design | pragmods is 1 critical trial per person; 5–10× more data per dollar; the pilot's median was 2.0 min for 12 choices | **differs:** in main everyone answers every stimulus | `rsa_live.yaml` `trials` |
| 25 | 10-09 → 10-11 | **200 people, 52 displays, 50 responses per display** per experiment (40 displays until 10-11) | power is flat in N and D (0.76–0.79); more displays help downstream: CV folds, clustered SEs, agents' conditions | **differs:** main uses 64 stimuli × N | `rsa_live.yaml` |
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
| 43 | 10-09 → 10-11 | **$0.50 a person, listed at 3 min** ($10/h; whole minutes, as Prolific takes them); about **$1,200** for 3 chains × 3 experiments × 200, $399 for experiment 1. The pilot (20 people) measured a 2.0-min median for 12 choices; the welcome screen says about 3 minutes | PI after the pilot (was $0.80 for an estimated 4 min) | | `rsa_live.yaml`, `PILOT_REPORT.md` §6 |
| 44 | 10-10 | **Exclude only earlier RSA participants** | subjective randomness is a different enough task (PI) | **differs:** main excludes every earlier auto-psych study | `exclude_earlier_participants_from: project` |
| 45 | 10-10 | Prolific ids only in each cell's `private/`, never copied off Sherlock; trial data stay on Sherlock until de-identified sharing is decided | privacy | same principle | `HANDOFF_live.md` |
| 46 | 10-10 | Live recruitment double-gated (config + flag) plus a typed "yes"; stages: test deploy → pilot → campaign, each with PI go-ahead | real money | same | `HANDOFF_live.md` |
| 47 | 10-10 | Server-assigned trial lists (`/assign`, round robin in arrival order); `/results?format=json`; the page posts trials as one JSON string | balanced lists; Firestore refuses nested arrays (stage 1) | RSA only (additive functions) | `functions/`, `src/rsa/live/` |
| 48 | 10-09 | **One core per process** (fits, agents, the harness) and `OPENBLAS_NUM_THREADS=1` | Research Computing flagged 140–160 threads per process | same aim as main's BLAS pinning; RSA also covers JAX | `src/rsa/cpus.py`, `HANDOFF_threads.md` |
| 49 | 10-09 | **Dense mass matrix** in NUTS for the live loop, and JAX's persistent compile cache | same posteriors, 2.5–9× fewer sampling steps (494 s → 149 s on the slowest model); repeated compiles read back | RSA only | `PLAN.md` "Compute" |
| 50 | 10-10 | Agent spend recorded per inner loop and per run | rehearsal 1 recorded none | same | `token_usage_summary.json` |
| 51 | 10-11 | **A candidate whose hypothesis repeats an existing model's word for word is rejected** (with the reason; the usual repair follows) | rehearsal 3's round-1 winner copied its parent's hypothesis, so the ledger, the briefs and the next critique described it without its new mechanism | RSA only | `RSALoop._same_hypothesis` |
| 52 | 10-11 | **The campaign runs experiment by experiment:** experiment 1 of all three chains first, reviewed, then the chains are resumed in place for experiments 2–3 (the experiment count is operational; finished stages are kept) | catch real-data surprises before spending the rest | RSA only | `OuterRun.run`, `HANDOFF_live.md` §4 |
| 53 | 10-11 | **Code freeze** for the campaign (the commit in §4): after it, only bug fixes, each logged here. The staging step refuses to resume a campaign on other code | claim 2 is a prospective test: the method is fixed before the data | same (main's sweeps run on one code) | `stage_code.sh` |
| 54 | 10-11 | **Bug fix:** the launcher's launch-cost print was a Python SyntaxError, so every `N_EXPERIMENTS` launch exited before its prompt (chain 0's experiment 1 was submitted by hand). The print moved into `src.rsa.live.config` | a bug | — | `63d9b0bd` |
| 55 | 10-11 | **Bug fix: catch trials are checked, not modelled.** Each experiment's `data/modelled.csv` is its responses without catch trials; the next design's data, the inner loop and the prospective score read it. Chain 0's experiment 1 (already run) fitted and critiqued its 370 catch trials | after exclusion they are all correct by construction: in the fits they pushed lapse rates down; the critic's catch-accuracy statistics measured the exclusion rule (z +2.2 to +3.4) | RSA only | `0ebe8de2`, `OuterRun.modelled` |
| 56 | 10-11 | **Bug fix:** the copy check (#51) also rejects a hypothesis wholly contained in an admitted model's | a refinement wrote only its parent's first sentence and reached chain 0's exported live set (`limited_attention_listener_2`) | RSA only | `8cafee80` |
| 57 | 10-11 | **Logged bug fixes reach a frozen work root only by a recorded re-stage:** `RESTAGE_REASON=<entry>` re-stages from a clean checkout and appends old → new and the reason to `$WORK_ROOT/code_history`; without it, other code is refused as before | the freeze pinned an exact commit, so no fix could be applied at all | same aim as main's runbook (rsync fixed code, record the commit) | `80a1c1df`, `9c4a9b37` (a re-stage compares contents: same-size files written within one second were skipped), `stage_code.sh` |

### 2.8 The final experiment (pre-registered)

| # | Date | Decision | Why | vs main | Where |
|---|---|---|---|---|---|
| 58 | 10-15 | **One pre-registered prospective test of the three finals; no lesions** | the primary goal is to tell the finals apart from the seeds and the other candidates; lesion tests of every accrued mechanism were too much (PI) | RSA only | `PREREG_FINAL.md` |
| 59 | 10-15 | **The bar is the best promoted seed** (chosen on the new data, after the fact, as in claim 2); one one-sided test per final, Holm across the three; the best starting model and every bar model reported beside it | PI | RSA only | `src.rsa.prereg.analysis` |
| 60 | 10-15 | **No secondary hypothesis** (chain 0's final beating the other two was dropped) | PI | — | `PREREG_FINAL.md` §2 |
| 61 | 10-15 | **Sized by power, no price cap:** frozen predictions from one pooled fit of all 20 models; displays half by the campaign's design rule (finals carried, bar at half the prior, quotas) and half random within the quotas; G and N from simulation (calibrated to claim 2's weakest experiment-3 advantage) and from the empirical √G scaling of claim 2's tests | "happy to go up in price here, so let's see what turns out" (PI) | RSA only | `src.rsa.prereg.power`, `PREREG_FINAL.md` §7 |
| 62 | 10-15 | **The pre-registration is frozen on OSF** as well as by the git freeze commit: the plan and the frozen files (`src.rsa.prereg.bundle`, hash-checked) registered **publicly** (no embargo) with OSF's Preregistration template before the study is published; the live launch waits for the approved registration | a public, timestamped record nobody on the project can edit (PI) | RSA only | `OSF_PREREG.md`; the live job refuses to publish without `osf_registration` |
| 63 | 10-15 | **G = 312 displays, N = 1,224 people** (~$816): the power rule's answer. The simulation is satisfied from G = 104; claim 2's empirical √G scaling (chain 2, experiment 3) needs G = 312 at ≥ 47 responses a display. G = 416 is impossible under the quotas (54 two-object displays) | PI, from `PREREG_FINAL.md` §7.3 | RSA only | `data/rsa/prereg/power.json`, `rsa_prereg.yaml` |
| 64 | 10-15 | **The pre-registration is frozen** (PI's go-ahead): design `bc7d60e5…` (156 EIG + 156 random, quotas met in both halves, 51 lists per display), predictions `6a0a476f…` (finals) and `704575376…` (bar), pooled data `19af0f54…`, code as at `507b5b9b`; simulated power of the frozen design ≥ 0.995 per final at δ = 0.0078. Next: the public OSF registration (PI), recorded as Addendum A | PI | RSA only | `PREREG_FINAL.md` §8, `rsa_prereg.yaml` |
| 65 | 10-15 | **The OSF registration links a tagged commit** (`rsa-prereg-final`, public repository) and quotes its full commit SHA and the frozen files' sha256, instead of uploaded files | OSF no longer takes files (its platform change; PI) | RSA only | `OSF_PREREG.md` |
| 66 | 10-15 | **Registered on OSF:** https://osf.io/s48jv (public, approved 10-15), linking tag `rsa-prereg-final` = `dbb38535…`. Its "Other" answer (commit SHA, file hashes) was left empty and the sample-size answer says "N": to be added by an OSF registration update before recruitment | PI | RSA only | `PREREG_FINAL.md` Addendum A, `rsa_prereg.yaml` `osf_registration` |
| 67 | 10-15 | **The pre-registered result** (OSF s48jv, updated and approved before launch; Prolific `6ad15ad5…`; 1,225 recruited, 1,108 kept, 14,404 trials): vs the best promoted seed `rsa_l2_singleton_distinct_feat_color_l0`, Holm at 0.05: chain 0's final **+197.9 ± 35.1 (z 5.64)** and chain 2's **+60.8 ± 22.9 (z 2.66)** reject; chain 1's +8.1 ± 24.1 (z 0.34) does not. All three beat the best starting model (z 8.0–10.3) and every bar model in point estimate. No deviation from the plan | the test | RSA only | `PREREG_RESULT_REPORT.md`, `PREREG_FINAL.md` Addendum B |
| 68 | 10-16 | **Consolidation (exploratory):** a superset of the three finals' mechanisms (13 switchable, 16 parameters; `src.rsa.consolidate`), backward elimination by grouped CV on the campaign's pooled data at a 2-SE rule fixed before any fit, then a prospective score of the lean model against chain 0's final on the pre-registered data | the leanest justifiable model of the phenomena (PI) | RSA only | `CONSOLIDATION.md` |
| 69 | 10-16 | **Memo twin of the superset, and a normalisation bug it found:** the superset's speaker and listener rewritten in memo (`memo_twin.py`), equal to the JAX version within float32 rounding (tested). It exposed `joint / (sum + 1e-10)` losing up to 0.0033 of probability under a peaked prior; fixed by computing the speaker and listener in log space (a first fix, `max(sum, 1e-30)` in 6c4dd826, made gradients NaN and is superseded), with tests for sums, twin equality and finite gradients. Elimination job 48051947 (old code) cancelled and restarted from scratch, before any result | the winner must translate to memo losslessly (PI) | RSA only | `CONSOLIDATION.md` §1 |
| 70 | 10-16 | **The elimination runs as one Slurm array per round on mcfrank and owners:** a driver replays the rule over saved CVs (`--mode plan`) and submits the round's missing models as workers (`--mode worker`), then queues itself. Same rule, folds, data and settings; tested to reach the single-process result | to have a result sooner, before a noon meeting (PI) | RSA only | `consolidate_array.sbatch`, `CONSOLIDATION.md` §4 |

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
| 10-10 | **Live stage 1 re-run: passed** | Sherlock 47272640, 49 min | chain 0's design with quotas, deployed; Prolific draft `6acaaa90…` | quotas met (6 two-object, 10 three-object, 13 mumble, 27 word); power 0.932 vs 0.931 free; page works end to end (consent → redirect; word, mumble and two-object displays; a list per id, the same on reload); data saved as one JSON string | the pilot can go (PI) | `STAGE1_FINDINGS.md` §9 |
| 10-10 | **Pilot** (live, 20 people) | Sherlock 47283766 + Prolific `6acabe15…`, 1 h 21 min | chain 0's experiment-1 design at N=20, $0.80 each | 20/20; median 119 s (80–670); 0 catch exclusions; every display exactly 5 responses; 1 timed out; Prolific held the 10 fastest for review (PI approved them) | **PI: $0.50 a person, estimate 2.5 min (Prolific takes whole minutes: see the report §6), welcome screen stays at 3 min; campaign about $1,200** | `PILOT_REPORT.md` |
| 10-10 | **Rehearsal 3** (critique) | Sherlock 47283774, 5 h 18 min, completed | chain 0, 1 experiment, 2 rounds × 6, rehearsal 2's ground truth | both rounds critiqued (8/8 statistics significant, none broken); 10 of 16 hypotheses cite a discrepancy; quotas met, power 0.933 vs 0.931 free; per-display KL 0.0168 → 0.0020; critique 15% of time and 17% of spend ($4.04 of $23.78) | **driver's call:** the critic scores all 51,530 trials (live 4.4%; 3 of 9 statistics pragmods-only) while selection ranks on live trials; 15/16 p at the floor (rank by z?); an incumbent with a stale hypothesis | `REHEARSAL3_REPORT.md`, `data/rsa/rehearsal3/` |
| 10-11 | Pre-campaign changes and **code freeze at `b162c960`** (later commits until the campaign ends change docs only, or are logged bug fixes) | cloud | the critic on live trials (#7); duplicate-hypothesis rejection (#51); 13 + 2 trials, 52 displays, $0.50 at 3 min (#24, 25, 43); resume with more experiments (#52) | fast suite 2,000 passed; slow end-to-end (two-experiment outer loop with the critique on live trials; loop with the critique on and off) passed | the campaign | this log, `HANDOFF_live.md` §4 |
| 10-11 | **Campaign, chain 0, experiment 1** | Sherlock 47331282_0, code b162c960, `$SCRATCH/auto-psych/rsa_campaign` | launched as 3 chains × 1 experiment (§4). **PI changed the plan:** his "experiment" is our chain and his "round" is our experiment. So it is chain 0 alone through experiments 1–3 (~$400), then chains 1 and 2 (~$800) if it works. Tasks 1 and 2 were cancelled in their design step, with no study created. Chain 0 runs experiment 1 (1 study, ~$133) and stops for review | | **`launch.sh` line 30 is a SyntaxError** (`\"` inside bash single quotes) whenever `N_EXPERIMENTS` is set, so the launcher exits before its prompt. Submitted with the launcher's own submit command by hand (`$WORK_ROOT/launch_campaign_e1.sh`). Fix before the next launch | `HANDOFF_live.md` §4 |
| 10-11 | **Campaign, chain 0, experiment 1: done** | Sherlock 47331282_0, 12 h 12 min; Prolific `6acb1fc4…` | 200 recruited, 185 kept (15 failed a catch trial); median 144 s at 15 choices; 24 held for review (56–89 s); quotas met, power 0.937 vs 0.942 free; inner loop 5 rounds, the best model changed every round; critic on live trials, 8/8 critiqued rounds, 31 of 34 hypotheses cite it | **best `candidate_load_codability_listener` +82.8 ± 22.0 nats on the live trials vs the starting incumbent** (and +101 / +87 / +60 / −5 / −8 on the existing sources); every starting model pruned by the guard; `live_vs` best seed +44.8 ± 54.9; agents $51, Prolific ~$133 | bug fixes #54–57 | `CAMPAIGN_C0_E1_REPORT.md`, `data/rsa/live/live_rsa_c0/` |
| 10-11/12 | **Campaign, chain 0, experiment 2** | Sherlock 47396066_0, re-staged b162c960 → e5fb029 (`code_history`, #54–57); Prolific `6acbe441…` | design ~2 h (fits the carried 8 first), power 0.753 (free 0.757), 21 models; 201 recruited in ~30 min, 193 kept; median 124 s; 39 held for speed. **Claim 2:** the committed `candidate_load_codability_listener` vs best seed **+126.9 (SE 37.4)**, vs best promoted +32.2 (SE 23.3). Inner loop: best → `prototype_typicality_bounded_listener` (rounds 3–5 unchanged), about +4 nats on live CV over the committed model; critiques 8, 8, 6, 8, 7 of 8 significant; $58.60 | | |
| 10-12 | **Chain 0, experiment 3: publish failed, recovered** | 47396066_0 FAILED 07:57 | the design finished (power 0.705); Prolific refused to publish: **account balance $125.33 < study cost $133.33**. A resumed run never publishes a recorded draft (the one-study guard), so after the PI topped up, the draft `6accf48f…` was published by hand (`src.runtime.prolific.publish_study`) and chain 0 resubmitted (47514554_0, same code, no re-stage); it collected from the recorded study | **Suggestion:** the launcher, or the deploy before creating a draft, should check that the balance covers the study | |
| 10-12 | **Chains 1 and 2 launched** (PI: chain 0's experiments 1–2 are enough to go on) | chain 1 47518520_1, e5fb029 | **staggered:** chain 2 goes in once chain 1's experiment-1 study has published, so chain 2's study (created after its ~1 h design) blocklists chain 1's participants. Launched together, both would publish at once and neither would exclude the other's. The chain 1 and 2 cells held only a seeded, undesigned experiment 1 from the tasks cancelled on 10-10; they resume in place | **Note:** chain 0's experiment 1 ran on b162c960 (catch trials modelled); chains 1 and 2 run wholly on e5fb029 | |
| 10-12 | **Chain 2's site flagged; chains 1–2 timing** | chain 2 → 47575675_2; chain 1 continuation 47575694_1 | chain 2's experiment-1 study was paused by Prolific at 196/200 after a participant's "malware warning": Google Safe Browsing flags `auto-psych-2c5da-rsa-c2.web.app` as deceptive (c0, c1 clean). The PI filed a false-positive report and a Search Console review (verification file deployed to the c2 site only). **Chain 2 runs experiment 1 only** (`N_EXPERIMENTS=1`) until the site is cleared or moved. **Rounds take 2–3.5 h**, so chains 1 and 2 need about 55 h for 3 experiments, over the 48 h limit. Chain 1 has a queued continuation (`--time=7-00:00:00 --dependency=afternotok:47518520_1`); the inner loop resumes from its last scored step | `SAFE_BROWSING_C2.md`. **For the driver:** move chain 2 to a new site, or wait for the review; `outer_live.sbatch`'s `--time=48:00:00` is too short for 3 experiments | |
| 10-13 | **Chain 0 complete** (experiments 1–3) | Sherlock 47514554_0 (experiment 3), e5fb029; Prolific `6accf48f…` | experiment 3: 201 recruited, 185 kept; median 135 s; power 0.705; 25 admitted, 15 rejected (**5 at the 30-min fit limit**, 8 not novel, 2 unconverged); best `boundary_avoidable_ambiguity_listener` (17 parameters) +47.5 ± 13.4 over the committed model on the cumulative live trials | **Claim 2, both tests:** the committed model vs best seed **+126.9 ± 37.4** (exp 2) and **+120.2 ± 28.0** (exp 3); vs best promoted seed +32.2 ± 23.3 and **+43.6 ± 14.4**; it beats all 17 bar models in both. Chain 0 cost $177 in agents + ~$400 Prolific | | `CAMPAIGN_C0_REPORT.md`, `data/rsa/live/live_rsa_c0/` |

## 5. Open

- **Rehearsal 3's questions,** decided 10-11: the critic scores the live trials (#7); a copied
  hypothesis is rejected (#51). Ranking by z instead of FDR survival is not needed: on the live
  trials alone the p-values carry information again (main's wording is kept).
- **Chain 0's experiment 1 was fitted with its catch trials** (before #55). Its export and
  `live_vs` stand as run; from experiment 2 on, no fit sees catch trials. To report.
- **Held submissions:** Prolific held 24 of 200 for being fast (56–89 s), but only 1 of them
  failed a catch trial; 14 of the 15 excluded were auto-approved. Someone must clear the held
  ones during each study.
- **Carry cap and families:** rehearsal 2 kept six close relatives and dropped two distinct
  models over the cap. One option is to fill the cap across families first. No change made.
- **A recovery test with an unfamiliar, pre-fixed ground truth:** optional, for the paper. It is
  simulated, so it can run alongside the live campaign.
- **The consent overlay** (main's deployment gate) is not in the page's accessibility tree:
  its "I agree" button is pointer-only (stage 1 re-run note).
- **The fit time limit filters complexity as the data grow** (chain 0, experiment 3: 5 of 15
  rejections at the 30-min admission limit; the winner has 17 parameters). Kept as frozen for
  chains 1–2, so the chains stay comparable; to report.
- **Design power falls across a chain** (0.94 → 0.75 → 0.71 in chain 0) as the carried models
  become close relatives: partly the loop converging, partly the carry-cap question above.
- **Wall time:** a chain of 3 experiments needs about 55 h, more than `outer_live.sbatch`'s 48 h.
  Handled without a code change: a continuation submitted with `--time` on the command line
  (Sherlock, chain 1). The script's default is left frozen.
- **Chain 2's Hosting site** is flagged by Google Safe Browsing (`SAFE_BROWSING_C2.md`); chain 2
  holds after experiment 1 until the review clears it or it moves to a new site.
- **Main's statistic timeout** (SIGALRM in `src/critique/ppc.py`) can fire inside a garbage-
  collector callback, where Python discards it (once in chain 0, experiment 3; the critique
  completed). An upstream issue in main's code; harmless here.
- **The campaign's finals** (pooled refit 8b34863c, `finals_pool_preds_final.json`):
  - **Shared by all three:** duplicates are less likely referents; a U-shaped prior over feature
    richness; a small rate of choosing objects the word doesn't fit.
  - **Two of three:** a preference for the sparsest matching object on word trials (chain 0's
    interior suppression; chain 1's "aspect" speaker goal).
  - **Chain 2 only:** a redundancy discount (0.50).
  - **In-sample on all 21,567 live responses,** chain 0's final fits best (chain 1's −160,
    chain 2's −189), and on each chain's own data. Its lead is on 4-object word trials; chain 1's
    prior is best on 3–4-object mumble trials.
- **Final experiment: DONE (#58–67).** Registered as OSF s48jv; 2 of 3 finals beat the best promoted seed on new people (chain 0 z 5.64, chain 2 z 2.66; chain 1 z 0.34). Earlier note: (`PREREG_FINAL.md`,
  code `src/rsa/prereg/`, Sherlock steps `HANDOFF_prereg.md`). Each final vs the best promoted
  seed on new people, frozen predictions, display-clustered SE, one-sided, Holm across 3; no
  secondary hypothesis, no lesions. Next: the 17 bar models' pooled fits (Sherlock), the power
  grid (driver), the PI's choice of G and N (the empirical scaling alone points to G = 312,
  N ≈ 1,250, ~$830), the design, the freeze, a test deploy, then live.

  Earlier plan, superseded: one pre-registered, prospective
  test of the three chains' final models, run after chains 1–2 finish:
  - **H1:** each final beats the best starting model and the best promoted seed on new people;
  - **H2:** each final needs every mechanism it accrued, tested by mechanically defined lesions
    (each mechanism's parameters fixed at their "off" value, the rest refitted).
  - **Fits:** all models fitted to all data so far, pooled across chains, frozen, then scored
    with no refit.
  - **Displays:** half stratified-random from the pool (generalization), half EIG over full vs
    lesioned models (power for H2).
  - **Size:** set by simulation.
  - **Scope:** the critic and agents play no part.
  - The draft pre-registration is to be written while chains 1–2 run.
- **Interim comparison of the chains' best models** (10-14; chain 0 after experiment 3, chain 1 after
  experiment 2, chain 2 after experiment 1):
  - **The prior is shared, but by ancestry:** singleton bonus 0.9–1.1, feature-count bonus
    0.16–0.34. Every chain was seeded with one of the singleton-prior models.
  - **Found independently in all three:** extremes of the feature range are favoured. Chain 0
    suppresses interior objects in the speaker; chain 1 has an extremity prior; chain 2 has an
    empty-object bonus that makes the prior U-shaped. No chain's seeds contain it.
  - **Two chains:** similarity-based confusion avoidance (chain 0 inherited it; chain 1 rebuilt it).
  - **Chain 2 only:** a redundancy discount (synonyms count as k^-0.64 alternatives).
  - **They differ in how people fail:** inattention that grows with display size (chain 0), word
    noise with a sharp speaker (chain 1), lapse only (chain 2).
  - **All three miss** duplicates (E9 twins) and Sikos 2021 experiment 3.
  - **Caveat:** the same LLM writes every chain's models, so cross-chain convergence mixes the data
    with the agents' own priors. The final experiment's lesions, organised by shared mechanism,
    test it.
  - **Pooled refit** (all three on the same data: existing plain + 8 live experiments, catch
    removed; `data/rsa/live/finals_pool_preds.json`, a8c8ee31):
    - **They are distinct hypotheses:** pool RMSE c0–c1 0.026, c0–c2 0.040, c1–c2 0.033, 3–5× the
      `rsa_l1`/`rsa_l2` gap.
    - **Where they disagree most:** a word fitting 3+ objects of graded feature richness. Chain 0
      puts 0.6–0.75 on the sparsest match and suppresses the intermediate object (its interior
      term); chain 2 spreads about a third onto the intermediate; chain 1 is in between. Chain 2
      also differs on mumble trials (its U-shaped prior).
    - **In-sample, chain 0's model fits every chain's live data best:**
      - chain 0's data: c1 −119, c2 −116 nats;
      - chain 1's: −15, −25;
      - chain 2's: −34, −27.

      This includes the data each chain selected its own model on. But it is in-sample, chain 0's
      model has about 5 more effective parameters and one more experiment of development, and
      chains 1–2 are unfinished. The final experiment decides it prospectively.
- **Known limits to report:**
  - simulated people are less noisy than real ones;
  - pruning on the live rows is weak;
  - valence and familiarization terms sit at their priors on plain data.
