# The live loop, as it runs (code `b6b7db7f`, rehearsal 2's settings)

What `src.rsa.outer.run` does, with every set of models and every scoring rule spelled
out. "Live" settings and rehearsal-2 settings are the same, except where a step says
*simulated*. Points worth reviewing are in §4, and the parameters in §5.

## 0. Notation

| Symbol | Meaning |
|---|---|
| **E** | the existing data: 50,587 human trials from 5 sources (pragmods, Sikos 2021, Mayn & Demberg 2022/2023/2026), train and test together |
| **L_k** | experiment *k*'s live trials (source `auto_psych`), after catch exclusion |
| **D_n** | the data before experiment *n*: E ∪ L_1 ∪ … ∪ L_(n−1) (`prior.csv`) |
| **C_n** | the data after it: D_n ∪ L_n (`cumulative.csv`) |
| **S** | the 5 starting models: `literal_listener`, `rsa_l1`, `rsa_l1_salience`, `rsa_l1_shared_prior`, `rsa_l2` |
| **P** | the 12 promoted seeds (10 group winners from run 2, plus `rsa_l2` and `rsa_l1`) |
| **K** | the chain's seeds, a subset of P. Chain 0 (the rehearsal) = {`competitor_confusion_speaker`, `isolated_graded_costly_l3`, `oddity_heuristic_listener`, `rsa_l2_singleton_feat_color_valence_l0`, `rsa_l2`, `rsa_l1`} |
| **B** | the *bar*: S ∪ P, the models claim 2 must beat |
| **M_n** | the models going into experiment *n* |
| **pool** | the 794 plain displays the design may pick: 3×3 and 4×4 games, twins, prior queries; no valence, familiarization or greyscale |

**Fit(m, D).** NUTS on D, dense mass matrix, 4 chains × (1,000 warm-up + 1,000 draws),
target acceptance 0.9, seed 0.

- **Convergence:** ≤ 0.1% divergent, R-hat ≤ 1.05, bulk ESS ≥ 100.
- **If not converged:** one refit at target 0.95 with seed + 1, and that refit is the fit.
- **Each fit** runs in its own process, pinned to one CPU.
- **Time limit:** only a *new candidate's* admission fit is limited (30 min). Every model
  already in play fits to completion.
- **Cache:** fits are cached by (model file, data file, settings).

## 1. Before experiment 1 (done once)

- **P:** run 2's 143 admitted real-data models, refitted on E, clustered into 10 groups
  by their predictions. Keep the best of each group by grouped CV, then add `rsa_l2` and
  `rsa_l1`.
- **The chains:**
  - `rsa_l2` and `rsa_l1` go into every chain;
  - the other 10 are dealt 4/3/3;
  - the split chosen is the one whose closest pair of models within a chain is farthest
    apart on the pool.
- **The ground truth (*simulated* only):** among run 2's unpromoted, converged models
  within 150 nats of the best on grouped CV over E, take the one farthest (pool RMSE)
  from its nearest chain-0 seed. It's fitted once, on E, and withheld from the agents.

## 2. Experiment *n*

### 2.1 The models going in, M_n

- **n = 1:** M_1 = K.
- **n > 1:** the previous inner loop's exported live set (§2.6), reduced to one model per
  distinct hypothesis (`carry.json`):
  1. **Order** the live set: trusted models first (converged, reliable LOO, CV converged,
     eligible), then by the loop's selection score (`sel_elpd`, §2.5).
  2. **Fit** each on C_(n−1) and take its posterior-mean choice-class probabilities on
     the pool.
  3. **Walk the order.** Keep a model unless its pool RMSE to a model already kept is
     < 0.002; then it's a twin, merged into that model. Stop keeping at **8**.

### 2.2 Design (`design/`)

1. **The design set** is M_n plus the bar B, in this order:
   - M_n first;
   - then each bar model, unless it is (a) the same file as a model already in, (b)
     within 0.002 pool RMSE of one, or (c) *simulated only*: the ground-truth file
     (left out, unnamed).
2. **Fit** each on D_n. Draw 200 posterior samples of its choice-class probabilities on
   every pool display.
3. **Prior over the design set:** 0.5 spread evenly over M_n, 0.5 spread evenly over the
   bar models that stayed in.
4. **Pick 40 displays by greedy joint EIG.** Each display gets 50 responses (200 people
   × 10 / 40).
   - **Scenarios:** 2,000; draw a model from the prior, one of its draws, and multinomial
     responses.
   - **Leave-one-out:** each scenario's likelihood average leaves out the draw that
     generated it.
   - **Stopping:** greedy picks stop at the noise floor; the remaining slots are filled
     by single-response EIG.
5. **Trial lists:** 200, one per participant. Each list is 10 designed displays (a
   balanced subset: each display is seen by 50 people), plus 2 catch trials and 1
   practice trial.
   - Every trial gets a random item (faces, sundaes, …), random words and a random
     screen order.
   - The data row records the designed display, so none of that randomization reaches a
     model.

### 2.3 Collect (`data/`)

- ***Simulated:*** each simulated person answers their list from the ground truth's
  posterior-mean probabilities, fitted on E. They are the same for every person and
  every experiment.
  - Each answer goes through the page's own records and `convert`, exactly as live data
    will.
- ***Live:*** the deployed page and Prolific (not wired yet).
- **Exclusion:** anyone who misses **any** catch trial is excluded. In rehearsal 1 this
  was 13 of 200, from the fitted lapse rate alone.
- **Participant ids** continue across experiments.

### 2.4 Prospective score, claim 2 (`private/…/prospective.json`)

1. **Fit on D_n:** every model in M_n (label: its name), every s ∈ S (`seed:s`) and
   every p ∈ P (`promoted:p`).
   - The promoted set is scored from experiment 1, because the chain began with only part
     of P.
2. **Score** each on L_n: held-out log predictive density, nothing refitted.
3. **Report:**
   - `best_seed` = the best `seed:` model; `best_promoted` = the best `promoted:` model;
   - `best_live` = the best model in M_n;
   - `live_vs`: best_live − each bar, with the SE clustered by designed display.

### 2.5 The inner loop (`model_loop/`): on C_n

**Setup**
- Copy M_n in as seeds and fit each on C_n (no time limit). A seed that fails to fit is
  dropped, with a ledger line.
- **Grouped CV folds:** 5 folds of whole units, balanced in trials within each source.
  - A unit is a condition of a source's experiment. For multi-trial experiments it is a
    display, so each live display is a unit.
- **The ledger starts empty.** It does not carry the previous experiment's attempted
  hypotheses (see the review points).

**Rounds** r = 1 … 5, each with 6 agent slots:

- **Slot roles:**
  - 3 *explore* slots, each given the next of 11 lenses in rotation;
  - 2 *refine the incumbent* slots;
  - 1 *refine a model of the agent's choosing*, from the menu of live and pruned models.
- **What an agent gets:**
  - the brief: every live model's hypothesis, source and standing (including whether it
    is eligible);
  - the refinement menu;
  - the ledger of tried hypotheses;
  - the cumulative trials C_n: raw rows; the `auto_psych` rows are the live ones;
  - a self-check command.

  No network, sandboxed, one CPU each, Gemini Flash, 40-minute timeout.
- **A slot that writes no candidate** is re-spawned once.
- **Admission, in order:**
  1. files present;
  2. code gate (an import allowlist; no file reads, eval or similar);
  3. the model contract on the training displays and the novelty pool;
  4. fit on C_n within 30 minutes;
  5. converged;
  6. finite LOO;
  7. **novelty:** posterior-mean predictions ≥ 0.002 RMSE from every live model, on the
     *novelty pool*. That pool includes valence, familiarization and greyscale displays.
- **A rejected candidate** is re-spawned once with the reason (a repair); the repair is
  final.
- **Score the round** (the standing):
  - every live model's grouped CV on C_n: 5 fold fits per model, each fold fitted on the
    other 4, out-of-fold lpd per trial;
  - **eligible:** a model's CV on the *existing* rows (non-`auto_psych`) is within
    **4 × clustered dse** of the best model there;
  - **trusted:** reliable LOO, converged, CV converged, and eligible;
  - **sel_elpd:** the summed out-of-fold lpd over the **live rows only** (L_1 ∪ … ∪ L_n,
    every live experiment so far);
  - **the incumbent** is the trusted model with the highest sel_elpd.
- **Stop early** once 2 rounds in a row leave the incumbent unchanged.

**End of the inner loop**
1. **Prune ineligible models:** any whose existing-data CV is behind by more than 4 dse.
2. **Prune trusted models** behind the incumbent on the live rows by more than
   **2 × clustered dse** (sel_elpd, clustered by live display).
3. **Cap:** while more than 12 are live, retire the lowest (untrusted first, then lowest
   sel_elpd). The incumbent is never retired.
4. **Export:** `best_model` = the incumbent; `live` = the survivors.

### 2.6 Recovery (*simulated* only, `private/…/recovery.json`)

- **Before:** M_n fitted on D_n. **After:** the inner loop's live set fitted on C_n.
- **For each model:**
  - `kl_pool`: the mean KL(ground truth ‖ model) over the 794 pool displays;
  - `kl_design`: the same over this experiment's 40 displays;
  - `rmse_pool`.
- **Also reported:** the exported model's distances, and the closest model before and
  after.

## 3. What the run reports, and where

| Question | Where |
|---|---|
| Does the loop's output predict new people better than the literature's models? (claim 2) | `prospective.json`, `live_vs` |
| Did the loop move toward the truth? (*simulated*) | `recovery.json`, `kl_pool` / `kl_design` |
| What was carried, and what was merged | `experiment<n>/carry.json` |
| What the design aimed at | `design/eig.json`: `models`, `model_groups`, `prior`, `merged`, power |
| What the agents tried | `model_loop/attempted_hypotheses.jsonl`, `history.json` |
| Agent spend | `model_loop/token_usage_summary.json`, `<run>/token_usage_summary.json` |

## 4. Points worth reviewing before the run

1. **Claim 2's comparison is chosen after the fact on both sides.**
   - `best_live` is the best of ≤ 8 models going in, picked after seeing L_n. The bar
     is the best of ~17, also picked after seeing L_n.
   - A pre-registered version: the model the previous inner loop *exported* (chosen
     before L_n existed) against each bar model, and against the best bar (post hoc,
     so conservative).
   - Recording that alongside costs nothing.
2. **One of the 11 exploration lenses asks for framing mechanisms** ("favorite / least
   favorite", one object in colour). The live displays can't test those, so they only
   affect eligibility on E, and they bred rehearsal 1's `twin_*_valence` copies.
   - Option: drop that lens in the live phase.
3. **The in-loop novelty gate still uses the novelty pool** (with valence and
   familiarization displays).
   - Twins that differ only off the live displays are admitted. They are merged only at
     carry-forward, so they can use up agent slots and the cap of 12 within an
     experiment.
   - Option: in the live phase, also require 0.002 on the plain-display pool at
     admission.
4. **The ledger starts empty every experiment.** Experiment 2's agents don't see what
   was tried or pruned in experiment 1.
5. **Selection sums over every live experiment so far** (L_1 ∪ … ∪ L_n), not only the
   newest. That's the intended reading of "rank on the live trials", but it means later
   experiments weigh the earlier ones equally.
6. **The simulation is optimistic about noise:** every simulated person answers from the
   same probabilities. Real people differ, and the live data will be noisier per trial.
7. **Pruning on the live rows is weak:** 2 dse on 2,000–4,000 trials over 40–80
   displays. The carried cap of 8 does most of the narrowing.

## 5. Parameters (OuterConfig / LoopConfig defaults)

| | Value |
|---|---|
| experiments per run | 3 live (2 in rehearsal 2) |
| participants per experiment / designed displays | 200 / 40 (10 per person, plus 2 catch) |
| rounds / slots per round / stale-round stop | 5 / 6 / 2 |
| selection | guarded: rank on live rows; eligible within 4 dse on existing rows |
| prune / live cap / carried cap | 2 dse on live rows / 12 / 8 |
| novelty (in loop) / same-hypothesis (carry, design) | 0.002 RMSE on the novelty pool / 0.002 RMSE on the plain-display pool |
| design prior | 0.5 carried, 0.5 bar |
| sampler | dense mass, 4 × (1,000 + 1,000), target 0.9 (refit 0.95) |
| candidate fit limit | 30 min (models in play: none) |
| catch exclusion | any miss |
