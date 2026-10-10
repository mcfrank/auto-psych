# Updating `auto-psych-pnas` to the final runs

Written 2026-10-07 against `paper/auto-psych-pnas/` as of that day. Nothing in
the paper was edited.

**Line references**
- "Main" = `auto-psych.tex`.
- "SI" = `supporting-info.tex`.

**Data sources**
- **Recovery:** `data/results/final_recovery/` (17 complete cells).
- **Human series:** `data/results/human_experiment_sr-oct26/` (sr-oct26, runs 1–3).

**Analysis outputs**
- **Where they are:** the analyses below write to
  `data/results/human_experiment_sr-oct26/analysis/results/` and
  `analysis/fits/`.
- **How they were fit:** every MCMC fit uses the live series' settings: 3000
  draws, 2000 tune, 4 chains, target_accept 0.8.
  - Each model was fit at the first sampler seed that passed the pipeline's
    convergence gate: ≤ 0.1% divergent, R-hat ≤ 1.05, bulk ESS ≥ 100.
  - ELPD-LOO goes through `loo_diagnostics`, so it includes the 2026-10-07
    exact-trial fix (`943cdcc6`). For runs 1–2 that fix changes no value.

Contents:
1. What in the paper is out of date (results, then methods)
2. The new numbers
3. What could not be computed here
4. How the fits were run (all done)

---

## 1. What is out of date

### 1a. Results

The figures in `figures/` are current except the two below. The text and
tables mostly still describe the June runs.

| Where | Paper now | Status |
|---|---|---|
| Fig. 2 top (`holdout_combined_rmse.pdf`) | new final-recovery figure | current |
| Fig. 2 bottom (`impossible_combined_rmse.pdf`) | old alien sweep | pending (`final_impossible_1`, 3/4 in per you) |
| Fig. 3 (`human_fit_comparison_experiment3_runs.pdf`) | 3-run figure | current |
| SI ablation figures + `impossible_combined_no_inner_loop.csv` | old (pre-final config) sweeps | out of date: needs a new ablation sweep (§3) |
| SI `example-trial.png` | a 7-vs-8-flip pair from the old, different-length design | out of date: needs a screenshot of a current page |
| Main 125 / 136 caption and text | "best seed model … fit to all of the data" | now true: the figure's orange line is the all-data fit at every point (see §2.1) |
| Main 134 | "five independent replicates for the held-out seed models" | 5 for `motif_stack`, 4 for each of the other three (3 cells never ran) |
| Main 136 | "consistently matched the ground truth better" | holds in all 17 cells (§2.1) |
| Main 138, SI 196 | inner-loop ablation claims | from the old sweeps; also see methods note M10 |
| Main 149 | durations, 262/288 unique pairs, median lengths, side/heads/length/alternation biases | all out of date (§2.2) |
| Main 151 | replicate-by-replicate story ("only slightly outperformed the Bayesian diagnosticity seed in replicate 3", "two virtually tied models in replicate 2"), pairwise RMSE 0.092 | out of date (§2.3, §2.4) |
| Main 153–172, Table 1 | June models, old seed names, ~11k trials | out of date (§2.5) |
| Main 174 | "~11k trials", noise ceiling .80, "83% of the explainable variance", held-out refit summary | out of date (§2.5, §2.6) |
| Main 176–182 | Minkowski typicality, evidence accumulation, Bayesian diag. + balance | out of date: different winners (§2.7) |
| Main 188–190 | Minkowski example; "largely conservative revisions to the seed models" | out of date: the new winners are large person-level cue bundles (§2.7) |
| Main 196 | "participant averages rather than individual variation" | every leading live model is person-level |
| SI 171 | 3.5 min average, \$1 pay | pay is \$1.40 for an estimated 7 min; the average time needs Prolific (§3) |
| SI 206–237, Table S1 | June models | out of date (§2.6) |

### 1b. Methods

An audit of every methods sentence against the code and the runs. I
spot-checked the three new findings (M1–M3). Items carried over from
`methods_changes.md` were re-verified; the ones now fixed are listed at the
end.

**Design facts the paper should disclose (new; all verified):**

- **M1. Trial order was not randomized.** In every one of the 9 experiments,
  all participants saw the 64 pairs in one fixed order. That order is the
  greedy selection order: joint-EIG picks first, then fills.
  - The cause: the deployed pages iterate `STIMULI.forEach` with no shuffle,
    and `prompts/3_implement.md` says not to reorder.
  - Left/right *was* randomized (p = .5 per trial).
  - Disclose this, or treat it as a limitation (order effects are confounded
    with design position).
- **M2. Experiment 1 was essentially the same design in all three runs.**
  - The live design seed is fixed at 42, and experiment 1 uses the prior
    predictive.
  - Runs 2 and 3 share all 64 experiment-1 pairs. Run 1 shares 57 of 64 with
    them; the rest differ only in mirror-image ties.
  - So the replicates diverge only from experiment 2 on (participants aside).
- **M3. Each experiment-1 design had only 9 joint-EIG picks and 55
  single-response fills**, because the joint EIG saturated. Experiments 2–3
  had 64 joint-EIG picks each (the `source` field in `design/stimuli.json`).
- **M4. Run 3 experiment 3 has 41 people.** One completed all 64 trials but
  returned the study, and collection keeps every saved row; the analyses below
  include them. Run 1 has 118 distinct people across 120 slots, because two
  took experiments 1 and 2.
- **M5. A LOO bug dropped models during the live runs.**
  - Run 2 experiment 2 started with 2 of its 8 carried models: the other 6,
    experiment 1's winner among them, were dropped as "non-finite ELPD-LOO".
    The cause was PSIS returning NaN on mirror pairs.
  - Run 1 lost two experiment-1 proposals and one carried model to the same
    bug.
  - It was fixed on 2026-10-07 (`943cdcc6`), after every run. The recovery
    sweeps ran on the unfixed code too.
  - This is a disclosed deviation.
- **M6. Run 1 experiment 1 was re-pruned at 4 dse after first pruning at 2**
  (`experiment1/model_loop/repruned.json`). The series prunes at 4; recovery
  prunes at 2.

**Main text**

- **Main 97:**
  - "a design agent adds these stimuli … and deploys it" → there is no design
    agent. EIG selection is programmatic, an implementation agent writes the
    jsPsych page, and the pipeline deploys it and recruits.
  - "all models whose fit is close to that of the best-fitting model are added
    to the registry" → the models not pruned (within k × stimulus-clustered
    dse of the best trusted model), capped at 8, all carried, with a uniform
    prior.
- **Main 104 (Fig. 1 caption):**
  - "it both proposes and computes test statistics" → the critic proposes
    them and the pipeline computes them.
  - "The best-fitting model is carried forward" → the surviving set (≤ 8).
  - Delete "In practice, we have an additional theorist agent in the outer
    loop…": none exists.
  - Optional: say *joint* EIG.
- **Main 113, seed descriptions:**
  - (3) "window typicality … typical of a larger sequence viewed through a
    finite attentional window" → the seed is `finite_experience_occurrence`:
    log P(the string occurs at least once in 20 fair flips).
    - "Window typicality" is the old seed's name. The figures call this one
      "finite experience".
  - (4) local representativeness: add the periodic-template penalty.
  - Its own manifest calls it "theory-inspired" (K&T published no equation),
    so qualify "faithful to the accounts".
  - The names should match the figures: encoding difficulty, regularity
    detection, finite experience, local representativeness.
  - Typo: "they models".
- **Main 130/132, recovery setup (missing):**
  - Generating parameters are the families' `DEFAULT_PARAMS` with no
    participant variation and side_bias 0; alien models use β = 4.
  - 40 simulated participants per experiment.
  - The loop starts from the other three seeds (all four for the alien
    models).
  - No implementation agent or deployment runs.
- **Main 134, the RMSE pool:** "all possible pairs of stimuli of length up to
  eight" → every distinct same-length pair at lengths 1–8 (43,435), minus the
  pairs in that cell's training data, about 180–192 per cell (43,243–43,256
  scored).
  - "The best-fitting model" = the ELPD-LOO rank-0 model among models with
    reliable PSIS-LOO. Its prediction is the posterior-mean p_left, averaged
    over training participants for person-level models.
- **Main 145 (Fig. 3 caption):** "difference standard error" → it is ArviZ's
  trial-level dse. Say so, or also give the stimulus-clustered SE (§2.3).
- **Main 200–201:** the "NEW PARAGRAPH ON STIMULUS SELECTION" placeholder is
  still there.

**Materials and Methods**

- **M&M 216:** recovery runs only the theorist and critic; no implementation
  agent.
  - The committed `holdout_recovery_faithful.yaml` still says opencode/Gemini.
    The sweeps overrode it at launch (every `holdout.json` records
    `claude-opus-5-5`), so update the config or document the override.
- **M&M 222, "a pair of sequences, an anonymized participant index, and a
  trial index":** a model that binds `trial_index` is rejected at admission.
  → "a pair of sequences and, optionally, an anonymized participant index".
  Models see only the raw columns.
- **M&M 223, "typically includes a bias parameter":**
  - Every seed has `beta` ~ U(0.2, 12) and `side_bias` ~ U(−2, 2).
  - Agent-written models choose their own parameters.
  - Typo: "heirarchical".
- **M&M 228, design (missing):**
  - 64 pairs per experiment.
  - The **joint** EIG of the set, with k ~ Binomial(N = 40, p_left) per pair.
  - Greedy selection stops at the Monte Carlo noise floor, then single-response
    fills.
  - Prior-predictive draws for experiment 1; posterior-predictive draws from
    refits on all data so far for later experiments.
  - Person-level models are predicted as 40 new participants.
  - Also: the sentence names only the outer loop.
- **M&M 230:** IRB protocol #20009 appears nowhere in the repository or the
  consent template. Confirm it.
- **M&M 232, critic:** the pipeline computes each statistic on 1000
  posterior-predictive replicates (two-sided empirical p, plus a BH-FDR q,
  which candidates also see).
- **M&M 232, gates:**
  - Add the code/import gate, the data contract, the 30-min fit limit and the
    convergence thresholds.
  - Novelty = posterior-mean RMSE ≥ 0.002 on a separate 512-pair pool
    (lengths 4–8).
- **M&M 232, pruning:**
  - "All models whose ELPD is 2 dse worse … are removed" → k × the
    **stimulus-clustered** dse against the best *trusted* model, with k = 2 in
    recovery and **4 live**; untrusted fits are not pruned; then a cap of 8.
  - Delete the following "All models that are within 2 dse … added" sentence.
  - "in~\ref{app:more-details}" is missing "Appendix".
- **Missing everywhere: the LOO reliability rule.**
  - A fit is trusted when ≤ 1% of trials have high Pareto k, with
    constant-likelihood trials exempt and scored exactly.
  - This rule, not ArviZ's any-k > 0.7 warning, decides export, pruning and
    the critic's incumbent.

**SI**

- **SI 65:**
  - Only experiment 1 starts from the seeds.
  - There is no posterior over models across experiments, only a uniform prior
    over the carried set.
- **SI 67:**
  - "ADD DETAILS OF EIG." placeholder.
  - The "single, distinct hypothesis" sentence belongs in the theorist
    paragraph (SI 164).
- **SI 69–73:** replace the single-response EIG(c) with the joint-set EIG,
  then describe:
  - **Selection:** greedy picks, the noise-floor stop and single-response
    fills.
  - **Leave-one-out:** each scenario's likelihood average leaves out the
    generating draw.
  - **Lazy search:** float32, validated against exact greedy.
  - **Draws:** 200 predictive draws per model and 1000 scenarios.
  - **Design-time refits:** 500/500/2 chains at target_accept 0.9 (not the
    model-loop settings).
  - **Person-level models:** predicted as new participants.
  - **Screening:** models that cannot be predicted are screened out on record
    (empty in all 9 live designs).
- **SI 75:** the *pipeline* deploys, recruits and polls, not the agent.
- **Algorithm 1:**
  - Remove the outer-loop `Theorist` step.
  - C = all same-length pairs at lengths 2–8.
  - Greedy joint EIG with Binomial outcomes.
  - M ← the surviving set, π ← Uniform(M).
  - "posterior π" → prior.
  - `M` → `\mathcal{M}`.
- **Algorithm 2:**
  - Remove λ and `− λ·length(m)`; the complexity penalty is 0, and the softmax
    is only reported.
  - m† = the rank-0 model among reliable fits, not argmax π.
  - Critic: J = 8, B = 1000, plus q; one retry, then no critique; a 5 s per
    call limit and a 300 s budget.
  - Candidates:
    - slot roles;
    - the inputs are ranked models (ELPD diff ± clustered dse), the ledger or
      refinement menu, and the critiques;
    - concurrent fits, then sequential admission;
    - the full admission gates;
    - retry and repair.
  - After the R rounds: prune, cap at 8, carry the survivors.
- **SI 159, MCMC:**
  - Recovery is **1000 tune / 1000 draws / 4 chains** (not 1000/2000/4) at
    target_accept 0.8. Human 2000/3000/4 is correct.
  - A model's declared target_accept is a floor (`motif_stack` declares 0.9),
    and a near-miss fit is refit once at 0.95.
  - Delete the 0.05-per-line complexity penalty and the softmax-posterior
    sentence.
  - Add the reliability rule.
- **SI 161–162:** the critic only writes `test_statistic(df)`; the pipeline
  simulates and evaluates.
- **SI 164:**
  - "Discouraged from stitching together" applies to exploratory slots only;
    refinement slots may graft one component as a single stated change.
  - Add the 11 exploration lenses, the refinement menu and the hypothesis
    ledger.
- **SI 169–171:**
  - **Pay:** \$1.40 for an estimated 7 min (\$12/hr) in all 9 manifests.
  - **Filters:** add **desktop only**; the US, English and ≥ 98% approval
    filters are confirmed.
  - **Earlier participants:**
    - "We did not exclude participants from previous studies" is true only for
      run 1 (2 people took two of its experiments).
    - Runs 2–3 excluded everyone from every earlier pipeline study.
  - **Missing:**
    - side randomization;
    - fixed trial order (M1);
    - the consent gate;
    - 41 participants in run 3 experiment 3.
- **SI 177–181, instructions:** two sentences changed.
  - → "The two sequences will have the same length."
  - → "You will complete 64 trials, which takes about 7 minutes."
- **SI prompts:**
  - `implement.txt`, `critic.txt` and `theorist.txt` are byte-identical to the
    code at every run's commit.
  - Exception: run 1 experiment 1's loop ran on a theorist prompt without the
    "Participant effects" section.
  - Say that the theorist's prompt also appends `CONTEXT.md`, the role brief,
    `existing_hypotheses.md`, the ledger or menu, and the critiques
    (`_build_candidate_prompt`).
- **New methods not described anywhere:**
  - **Agent sandbox:** bubblewrap, a read-only tree of only `src/` and the run
    tree, and an allowlisted environment.
    - Agents **did have network access**; the activity logs show only docs and
      arXiv links.
  - **Leak check:** the ground-truth-name scan.
  - **The hypothesis ledger.**
  - **The LOO reliability rule.**
- **Template leftovers:**
  - significance statement, contributions and competing interests;
  - the corresponding author inside `\equalauthors`;
  - SI title and authors;
  - `\instructionspage`.

**M10. The inner-loop ablation, under the current definitions.**
- With `INNER_LOOP_ITERATIONS=0` no models are proposed. The "best model" is
  then the ELPD-best reliable seed, which is essentially the orange baseline.
- So SI 196's "sometimes found models that matched the ground truth better
  than the seed models" can no longer happen.
- A rerun has to say what the ablation's best model is.

**Fixed since the 2026-10-02 audit:**
- the agent backend (Opus 5.5 everywhere);
- the seed names in Main 113;
- Main 218's "Appendix";
- the Fig. 1 PDF labels;
- the prompts still match the code.

---

## 2. New numbers

### 2.1 Model recovery (Fig. 2 top; Main 129–138)

**The cells:**
- 17 complete cells: `motif_stack` × 5 and the other three × 4.
- Each is 3 experiments × (seed step + 5 rounds of 6 candidates), with Opus 5.5
  agents.
- Recovery MCMC: 1000 tune / 1000 draws / 4 chains.

**The comparison:**
- **RMSE pool:** every same-length pair at lengths 1–8 not used in training,
  43,243–43,256 pairs per cell.
- **Baseline:** the ELPD-best other seed, fit to all of the cell's data.

| Ground truth (figure name) | n | final best-model RMSE | best-other-seed RMSE | ratio | final r | seed r |
|---|---|---|---|---|---|---|
| encoding difficulty (`falk_konold_dp`) | 4 | 0.026 ± 0.006 | 0.046 ± 0.002 | 0.56 | 0.994 | 0.984 |
| finite experience (`finite_experience_occurrence`) | 4 | 0.023 ± 0.003 | 0.077 ± 0.006 | 0.30 | 0.955 | 0.534 |
| local representativeness | 4 | 0.068 ± 0.008 | 0.130 ± 0.000 | 0.52 | 0.880 | 0.441 |
| regularity detection (`motif_stack`) | 5 | 0.094 ± 0.013 | 0.175 ± 0.010 | 0.53 | 0.940 | 0.802 |

(± = SEM across replicates.)

**Per-cell results:**
- In **17 of 17 cells** the final best model beats that cell's best other seed,
  on both RMSE and Pearson r.
- In 17 of 17 cells the final best model is a discovered model.
- The incumbent changed 15.1 times per cell on average (range 12–17, out of 18
  scoring steps).

For `motif_stack` the best model's RMSE gets *worse* during experiment 1
before improving (it peaks near 0.19 at round 2). In all other ground truths
it improves within experiment 1.

### 2.2 Human experiments: descriptives (Main 149)

From `describe_run.py`'s definitions, applied to all three runs.

**Size:**
- 3 runs × 3 experiments.
- 40 participants per experiment (41 in run 3 experiment 3).
- 361 participant-experiments, 359 distinct people, 64 trials each.
- **23,104 trials** (7,680 + 7,680 + 7,744).

**Collection time:** first Prolific poll to the target, per experiment
- run 1: 29.2, 30.8, 31.8 min;
- run 2: 26.7, 53.0, 28.2 min;
- run 3: 15.6, 36.8, 15.1 min.
- → **15–53 minutes** (the paper says 25–48).

**Stimulus reuse:** 9 × 64 = **576** pairs shown (not 288).
- 416 distinct unordered pairs: **72% unique**.
- 339 distinct after merging H/T relabellings (the definition `describe_run.py`
  uses for the September figure): **59%**.
- 67 of those 339 appear in more than one experiment.
- Much of the reuse comes from M2: runs 2 and 3 share all 64 experiment-1
  pairs, and run 1 shares 57 of them.

**Lengths:**
- All pairs are same-length.
- 453 of 576 pairs are 8 flips long.
- Pair lengths: 2: 12, 3: 30, 4: 19, 5: 15, 6: 2, 7: 45, 8: 453.
- The median sequence length is **8 in every experiment** (the old text says
  6–8).

**Biases:** the proportion of a participant's choices following each rule;
mean over participant-experiments with a t 95% CI, n = 361.

| Bias | All runs | run 1 | run 2 | run 3 | Paper (June) |
|---|---|---|---|---|---|
| side (chose left) | 0.503 [0.495, 0.511] | 0.509 | 0.504 | 0.497 | 0.50 [0.49, 0.51] |
| more heads | 0.501 [0.493, 0.509] | 0.515 | 0.501 | 0.488 | 0.52 [0.51, 0.54] |
| longer sequence | — (no different-length pairs) | — | — | — | 0.57 [0.55, 0.60] |
| more alternations | **0.658 [0.639, 0.677]** | 0.656 | 0.687 | 0.630 | 0.62 [0.60, 0.64] |

So: no side bias, **no heads bias**, a length bias that can't be measured
(the sentence should go), and a stronger preference for more alternation.

### 2.3 Fig. 3: each run's best model vs. the seeds (Main 145–151)

Each run's experiment-3 winner and the four seeds, all refit on that run's full
data. Values are ELPD-LOO relative to the winner, ± the trial-level dse used in
the figure, with the stimulus-clustered SE in brackets.

| | run 1 (7,680 trials) | run 2 (7,680) | run 3 (7,744) |
|---|---|---|---|
| best agent model | `personal_pattern_detection_gain` (−3450.2) | `edge_streak_primacy_recency_pattern_rule` (−3259.7) | `mild_symmetric_trick_coin_streak_weary` (−3559.1) |
| local representativeness | −1088.5 ± 41.7 [60.3] | −1125.7 ± 41.5 [67.6] | −935.0 ± 38.6 [71.0] |
| regularity detection | −1158.5 ± 42.5 [60.7] | −1346.9 ± 43.6 [91.8] | −1118.6 ± 40.4 [85.5] |
| encoding difficulty | −1415.5 ± 43.8 [69.3] | −1534.8 ± 44.3 [95.0] | −1361.9 ± 41.6 [93.1] |
| finite experience | −1776.8 ± 44.7 [98.0] | −1669.6 ± 43.6 [109.9] | −1453.9 ± 42.4 [113.9] |

**How large the gaps are:**
- In every run the winner beats the best seed (always local representativeness)
  by ≥ 935 nats: 24–26 trial-level SEs, 13–18 clustered SEs.
- The seed ranking is identical in all three runs.
- **Per trial:** the typical (geometric-mean) probability of the observed
  choice is 0.64, 0.65 and 0.63 for the winners, against 0.55, 0.57 and 0.56
  for the best seed (chance is 0.50).

**Replacement for Main 151's story:**
- Each replicate's winner beats every seed decisively; none is a near-tie with
  a seed.
- **Within the loop**, each winner is not clearly separated from its closest
  rival (loop ELPD differences, ± dse):
  - run 1: the top four are within 0.6 nats;
  - run 2: the runner-up is 2.3 ± 3.4 behind;
  - run 3: the runner-up is 1.4 ± 4.1 behind, and all eight carried models are
    within 10 nats.
- The three runs chose a different winner in every experiment.

**One caveat:** run 2's winner is the seed-2 fit (seed 6 in the generalization
record, not in the local cache). The two ELPDs differ by 1.3 nats.

### 2.4 How similar are the winners? (Main 151, the "0.092")

Population-level predictions on the loop's 512-pair novelty pool (lengths 4–8),
with person-level models averaged over their run's participants. This is the
method `report_fits.py` used for run 1's 0.092 / 0.029. Each model is as fit
on its own run.
(`results/winner_pool_rmse_runs123.json`)

| Pair | RMSE |
|---|---|
| run 1 winner vs run 2 winner | **0.094** (the maximum) |
| run 1 winner vs run 3 winner | 0.090 |
| run 2 winner vs run 3 winner | 0.066 |
| *for scale:* local representativeness fit to run 1 vs run 2 / run 1 vs run 3 / run 2 vs run 3 | 0.034 / 0.045 / 0.034 |
| *for scale:* each winner vs its run's local representativeness fit | 0.101 / 0.065 / 0.077 |

**What it says:**
- **Main 151:** "the maximum pairwise RMSE between [the three] best-fitting
  models … was 0.094".
- The winners differ from each other about as much as each differs from the
  best seed. They agree less than the old text's "similar predictions"
  implies.
- The old "0.029 between the two tied replicate-2 models" has no counterpart:
  I didn't fit the runners-up.

### 2.5 Table 1: all runs pooled (Main 153–174)

**Fits:** 23,104 trials, 359 people (`results/paper_table_pooled_runs123.csv`).
- Every fit converged and passes the LOO reliability rule.
- Seeds: 20261003, except `motif_stack` and run 2's winner at seed 6. Run 2's
  winner failed at 20261003 (3,003 divergent transitions, R-hat 1.53).

| Model | Type | k | ELPD-LOO | ΔELPD | SE(Δ) | SE_stim(Δ) | RMSE | R² |
|---|---|---|---|---|---|---|---|---|
| `personal_pattern_detection_gain` | Win (run 1) | 17 + 6 person-level | **−10180.4** | **0.0** | 0.0 | 0.0 | **0.074** | **0.910** |
| `edge_streak_primacy_recency_pattern_rule` | Win (run 2) | 15 + 6 person-level | −10422.5 | 242.1 | 36.2 | 48.5 | 0.085 | 0.880 |
| `mild_symmetric_trick_coin_streak_weary` | Win (run 3) | 14 + 4 person-level | −10769.8 | 589.3 | 46.9 | 71.0 | 0.098 | 0.840 |
| local representativeness | Seed | 5 | −13439.1 | 3258.6 | 71.0 | 186.3 | 0.132 | 0.713 |
| regularity detection (`motif_stack`) | Seed | 7 | −13914.5 | 3734.1 | 74.3 | 217.8 | 0.164 | 0.558 |
| encoding difficulty (`falk_konold_dp`) | Seed | 2 | −14585.1 | 4404.7 | 75.0 | 256.7 | 0.190 | 0.403 |
| finite experience | Seed | 2 | −15188.7 | 5008.3 | 77.1 | 352.3 | 0.211 | 0.262 |

**What it says** (vs. the old Main 174 claims):
- **Winners vs. seeds:** all three winners beat all four seeds on all the data.
  The weakest winner (run 3's) is 2,669 nats ahead of the best seed (local
  representativeness). The best winner is 3,259 ahead, 17.5 clustered SEs.
  - The old text had "replicate 3 slightly outperformed the best seed"; now
    every replicate's winner is far ahead.
- **Winners vs. each other:** they are clearly distinguishable on the pooled
  data. Run 1's beats run 2's by 242 ± 48.5 (clustered), 5.0 SEs, and run 3's
  by 589 ± 71.0, 8.3 SEs.
- **Explainable variance:** the best model's per-stimulus R² of 0.910 is
  **98% of the explainable variance** (noise ceiling 0.930; the old text had
  83% of .80). The best seed explains 0.713 (77%).
  - These R² values are in-sample, with person-level fits. The R² of a
    population-level prediction for *new* people would be lower; see the
    caveats under §2.6.
- **Seed ranking:** local representativeness > regularity detection >
  encoding difficulty > finite experience, the same as in every run (Fig. 3).
- **Reliability:** the winners' maximum Pareto k is 2.15, 1.62 and 0.79. All
  three pass the pipeline's rule (≤ 1% of trials with high k).
- **k:** the person-level vectors have 400 slots each; 359 are used here.

**Design:**
- The four seeds and the three winners are each fit on all 23,104 trials.
- Participant ids are offset per run, giving 359 people.
- The per-stimulus RMSE and R² use in-sample predictions, as `report_fits.py`
  does (person-level models: each person's own fit). The stimulus is the
  unordered pair, 416 of them.

**Noise ceiling (pooled):**
- Spearman–Brown split-half reliability **0.930** (split-half r = 0.869);
  variance-decomposition reliability 0.929.
- Per run: 0.932, 0.932 and 0.935.
- The paper says .80.

### 2.6 Table S1: each winner on the two runs that did not produce it (SI 206–237)

**Design:**
- For each run *r*, its winner and the four seeds are fit on the pooled data
  of the other two runs (15,360–15,424 trials).
- The noise ceilings of those datasets are 0.930, 0.932 and 0.933.

**Status:** all three blocks are done. Every fit converged and passes the
pipeline's LOO reliability rule. Two winners needed several sampler seeds:
- **Run 3's winner on runs 1+2:** failed at 20261003 (R-hat 1.53), converged
  at seed 6.
- **Run 2's winner on runs 1+3:** failed at eight seeds before converging at
  seed 8, the ninth attempt.
  - Mostly it was one chain stuck for the whole run (3,000 of 12,000
    transitions divergent, R-hat ≈ 1.5–1.7). Two attempts mixed slowly
    instead (R-hat 1.14 and 1.16).
  - That's worth knowing about this model's geometry, though the converged
    fit is fine.

**Block 1:** held-out run 1, i.e. data = runs 2+3; 15,424 trials, 241 people.
(`results/paper_table_without_run1.csv`)

| Model | Type | k | ELPD-LOO | ΔELPD | SE(Δ) | SE_stim(Δ) | RMSE | R² |
|---|---|---|---|---|---|---|---|---|
| `personal_pattern_detection_gain` | Win (run 1) | 17 + 6 person-level | **−6684.7** | **0.0** | 0.0 | 0.0 | **0.073** | **0.913** |
| local representativeness | Seed | 5 | −8889.8 | 2205.1 | 57.3 | 128.3 | 0.123 | 0.750 |
| regularity detection (`motif_stack`) | Seed | 7 | −9291.5 | 2606.8 | 60.4 | 162.0 | 0.162 | 0.565 |
| encoding difficulty (`falk_konold_dp`) | Seed | 2 | −9714.4 | 3029.8 | 60.6 | 183.3 | 0.187 | 0.420 |
| finite experience | Seed | 2 | −9942.0 | 3257.4 | 62.3 | 234.4 | 0.203 | 0.316 |

**Block 2:** held-out run 2, i.e. data = runs 1+3; 15,424 trials, 239 people.
(`results/paper_table_without_run2.csv`)

| Model | Type | k | ELPD-LOO | ΔELPD | SE(Δ) | SE_stim(Δ) | RMSE | R² |
|---|---|---|---|---|---|---|---|---|
| `edge_streak_primacy_recency_pattern_rule` | Win (run 2) | 15 + 6 person-level | **−7150.0** | **0.0** | 0.0 | 0.0 | **0.087** | **0.879** |
| local representativeness | Seed | 5 | −9046.6 | 1896.7 | 56.5 | 121.4 | 0.138 | 0.699 |
| regularity detection (`motif_stack`) | Seed | 7 | −9293.4 | 2143.4 | 60.5 | 133.8 | 0.163 | 0.576 |
| encoding difficulty (`falk_konold_dp`) | Seed | 2 | −9792.4 | 2642.5 | 62.1 | 160.7 | 0.197 | 0.381 |
| finite experience | Seed | 2 | −10252.5 | 3102.5 | 62.7 | 227.9 | 0.224 | 0.205 |

The winner's maximum Pareto k is 1.68, but fewer than 1% of its trials have a
high k.

**Block 3:** held-out run 3, i.e. data = runs 1+2; 15,360 trials, 238 people.
(`results/paper_table_without_run3.csv`)

| Model | Type | k | ELPD-LOO | ΔELPD | SE(Δ) | SE_stim(Δ) | RMSE | R² |
|---|---|---|---|---|---|---|---|---|
| `mild_symmetric_trick_coin_streak_weary` | Win (run 3) | 14 + 4 person-level | **−7214.7** | **0.0** | 0.0 | 0.0 | **0.105** | **0.826** |
| local representativeness | Seed | 5 | −8932.4 | 1717.6 | 55.3 | 125.8 | 0.129 | 0.739 |
| regularity detection (`motif_stack`) | Seed | 7 | −9235.6 | 2020.8 | 57.5 | 146.7 | 0.158 | 0.605 |
| encoding difficulty (`falk_konold_dp`) | Seed | 2 | −9662.0 | 2447.2 | 59.3 | 167.3 | 0.191 | 0.427 |
| finite experience | Seed | 2 | −10175.6 | 2960.9 | 60.7 | 217.8 | 0.225 | 0.204 |

All five fits in block 3 are reliable; the winner's maximum Pareto k is 0.73.

**Comparing the blocks:**
- **Winner vs. best seed:** in every block the winner, refit on the two runs
  that did not produce it, beats every seed decisively. Over the best seed
  (local representativeness) the margins are:

  | Block | ΔELPD | Clustered SEs |
  |---|---|---|
  | run 1's winner | 2,205 | 17.2 |
  | run 2's winner | 1,897 | 15.6 |
  | run 3's winner | 1,718 | 13.7 |

  - The old SI text ("replicate 3 performed slightly better than the seed
    models") no longer applies: all three are far ahead.
- **Per-stimulus RMSE**, winner vs. best seed: 0.073 vs 0.123 (run 1),
  0.087 vs 0.138 (run 2), 0.105 vs 0.129 (run 3). Run 3's winner gains the
  least.
- **Seed ranking:** local representativeness > regularity detection >
  encoding difficulty > finite experience in every block, the same as in
  Fig. 3 and Table 1.

**How to read it:**
- **What "held out" means:** the winner was *discovered* on run 1 and *refit*
  on runs 2+3. Its parameters, person-level ones included, are fit to those
  people, so it is held out only in the sense of the old Table S1.
- **Reliability:** all five fits are reliable by the pipeline's rule. The
  winner's maximum Pareto k is 1.58, but fewer than 1% of its trials have a
  high k.
- **RMSE and R²:** in-sample, as in the old tables. A person-level model's are
  helped by its 6 × 241 person parameters, so they are not comparable
  like-for-like with the seeds'. The ELPD-LOO gap (> 2,200 nats, 17
  clustered SEs) is the fairer comparison, though trial-level LOO also favors
  models that fit known people.
- **Generalization to new people:** for that, see the generalization and
  leave-one-participant-out analyses in `analysis/` (`generalization_analysis.py`,
  `lopo_analysis.py`). They exist for runs 1–2 only.

### 2.7 The winners, for the qualitative section (Main 176–190)

All three are **person-level**: they have person-specific parameters (400
slots) for some of sensitivity, ideal switching rate, pattern sensitivity,
side habit and guessing. Run 1's has 17 population-level parameters and 6
person-level vectors. All three judge sequences by switching relative to a
preferred rate, plus penalties on streaks and on structure that looks
designed.

- **Run 1: `personal_pattern_detection_gain`.**
  - **Switching:** each sequence's switching rate is judged against a personal
    ideal, leniently toward over-alternation.
  - **Personal weights:** heads/tails imbalance and the longest streak.
  - **Designed-looking structure:** the final streak, mirror symmetry,
    complement symmetry, run-length variety and the most lopsided stretch.
  - **The one change:** a single personal "pattern-detection gain" scales all
    the structure cues together, plus a personal guessing rate.
- **Run 2: `edge_streak_primacy_recency_pattern_rule`.**
  - **Tally span:** the span of the running heads–tails tally, judged against
    a personal expected span.
  - **Switching:** closeness to a personal ideal switching rate.
  - **Pattern cues:** chunk variety and a "rule-built" penalty, both scaled by
    a personal pattern sensitivity.
  - **Side lean:** a personal left/right lean.
  - **Edge streaks:** shared penalties on every flip of the *final* run beyond
    the first (τ = 0.099 [0.056, 0.142]) and, the winning change, of the
    *opening* run (τ_open = 0.076 [0.029, 0.124]). People prefer sequences
    that start and end with a switch.
- **Run 3: `mild_symmetric_trick_coin_streak_weary`.**
  - A Bayesian "randomness as evidence" account, the same idea as the
    regularity-detection seed.
  - **Chance vs. regular processes:** a sequence looks random to the extent
    that the person's own model of a fair coin explains it better than
    regular processes: a switch-biased coin, a trick coin and a repeating
    motif with slips (slip rate ε = 0.05 [0.01, 0.10]).
  - **The person's picture of chance:**
    - It rarely switches twice in a row (d = −0.21 [−0.30, −0.12]).
    - A switch feels more due the longer a streak lasts (the gambler's
      fallacy, g = 0.10 [0.04, 0.16] per flip beyond two).
  - **Short sequences:** the evidence is diluted for short sequences.
  - **Differences between people:** a person-specific believed switch rate,
    motif weight, signed heavy-tailed sensitivity and side habit.
  - **The winning change:** the imagined trick coin is symmetric and only
    mildly rigged (κ = 12.3 [3.2, 23.3]; 2 would be uniform).
- **For the Discussion (Main 188–190):**
  - The winners are not "conservative revisions" of one seed. Each is a bundle
    of 4–6 person-level vectors and many shared cues. The best model changed
    13, 15 and 14 times in runs 1, 2 and 3, always to a discovered model
    (human README).
  - Run 3's is the closest to a literature account (regularity detection).
  - The "prototype vs. Bayesian" contrast in Main 182 no longer applies as
    written. Runs 1–2 are cue-based (switch rate vs. a personal ideal, plus
    structure penalties), and run 3 is Bayesian.

Posterior summaries of every population-level parameter:
`results/winner_parameters_runs123.json`.

---

## 3. Not computed here

- **Alien recovery (Fig. 2 bottom):** `final_impossible_1` was not in the
  repository's results; you have 3/4 elsewhere.
- **Inner-loop ablation (SI figures):** needs a new sweep on the final config,
  with `INNER_LOOP_ITERATIONS=0`. Also see M10 for what its best model means.
- **Average completion time (SI 171):** the collected files have no timing
  columns, so take it from Prolific's export.
- **Example-trial screenshot (SI 186):** take one from a deployed sr-oct26
  page.
- **IRB number (M&M 230):** not in the repository; confirm it.

---

## 4. Resuming the stopped fits

**Where everything is:**
- **Scripts:** `data/results/human_experiment_sr-oct26/analysis/paper_update/`.
- **Fits:** in the shared `analysis/mcmc_cache/`, each with its converged seed
  recorded in `analysis/fits/<dataset>/fit_seeds/<model>.json`. A rerun skips
  nothing automatically, but every finished fit is a cache hit.

**Fit seeds:**
- Each fit uses the first sampler seed of `SEEDS` that passes the convergence
  gate.
- `motif_stack` and run 2's winner start at seed 6: seed 20261003 failed for
  both on every dataset tried.
- Each attempt has a time limit: 45 min for a seed, 4 h for a winner.

**Update (2026-10-08):** every fit is done, and Table 1 and all of Table S1
are filled in. This section is only a record of how they were run.

**Commands:** three winner fits remained at the stop. Each needs roughly 1.5–4 h awake and
plugged in, and they can run together on 12 of the 15 cores.

```bash
P=data/results/human_experiment_sr-oct26/analysis/paper_update
printf "%s\n" "pooled_runs123 edge_streak_primacy_recency_pattern_rule" \
              "without_run2 edge_streak_primacy_recency_pattern_rule" \
              "without_run3 mild_symmetric_trick_coin_streak_weary" \
  | caffeinate -i xargs -P 3 -L 1 $P/run_one_fit.sh
# then build the tables (cache only, no MCMC):
PYTENSOR_FLAGS="cxx=$PWD/$P/clangwrap/clang++" \
  uv run python $P/paper_tables.py pooled_runs123 without_run2 without_run3
```

**After the fits:**
- `paper_tables.py` writes `analysis/results/paper_table_<dataset>.csv`.
- Then fill §2.5 and the other two blocks of §2.6.

**Notes:**
- `without_run3`'s winner will hit its failed seed-20261003 fit in the cache
  and move straight on to seed 6.
- On Sherlock, drop `clangwrap/`: it only works around Apple clang 21's removed
  `-ld64` flag.
