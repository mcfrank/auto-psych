# Pre-registration: do the campaign's final models predict new people better than the bar?

**Status: FROZEN 2026-10-15** (§8), to be **registered publicly on OSF** before any participant is
recruited (`OSF_PREREG.md`; PI 2026-10-15). After the freeze, this file changes only by dated
addenda at the end. The first records the OSF registration and the registered bundle's hash,
which cannot be in the files they register.

- **Authors:** Michael C. Frank (PI), with the auto-psych pipeline.
- **Drafted:** 2026-10-15.
- **Decisions:** PI, 2026-10-15 (`CAMPAIGN_LOG.md` #58–#61):
  - the bar is the best promoted seed;
  - no secondary hypothesis;
  - no lesions;
  - the size is set by the power simulation, with no fixed price cap.

## 1. Background

The RSA live campaign (`CAMPAIGN_LOG.md`, `CAMPAIGN_C{0,1,2}_REPORT.md`) ran three independent
chains, each of three live experiments of 200 people. In each experiment, an inner loop of coding
agents proposed models of how people resolve reference in simple displays, and the data chose
among them. Each chain ended with a final model:

| chain | final model | parameters |
|---|---|---|
| 0 | `boundary_avoidable_ambiguity_listener` | 17 |
| 1 | `feature_surprisal_ensemble_listener` | 9 |
| 2 | `competition_endpoint_binding_lapse_l2` | 13 |

The **bar** is the 17 models the campaign was measured against:
- the 5 starting models (`literal_listener`, `rsa_l1`, `rsa_l2`, `rsa_l1_salience`,
  `rsa_l1_shared_prior`);
- the 12 promoted seeds: the best models of the existing-data phase, one per cluster
  (`data/rsa/live_seeds/models`).

The campaign's claim-2 tests scored each chain's committed model on the next experiment's new
people. Against the best promoted seed, they were positive in 6 of 6 tests, and beyond 2 SE in 3
(chain 0's experiment 3, and both of chain 1's tests). Chain 2's two tests were +13.6 ± 27.2 and
+19.2 ± 15.1.

No final model has been tested on people it was not fitted to. This experiment tests them.

## 2. Hypotheses

**One primary hypothesis per final model f** (three tests):

> **H1_f:** on new participants, f's summed log predictive density (lpd) exceeds that of the
> best promoted seed.

- **Null hypothesis:** the difference is ≤ 0.
- **Test:** one-sided.
- **Multiple comparisons:** Holm's step-down procedure over the three tests, at familywise
  α = 0.05.

There are no secondary hypotheses. Everything in §6.3 is descriptive.

## 3. Models and their frozen predictions

### 3.1 The fit

Every model is fitted **once**, to the same pooled data, and its predictions are frozen. Nothing
is refitted to the new data.

**The pooled data:** 70,841 trials.
- The existing plain-display trials: the literature data the campaign fitted, the same in every
  chain.
- Every live experiment's `data/modelled.csv` (catch trials removed), from all three chains,
  with participant ids prefixed by chain.

**The fit:** the loop's own fit rule (`src.rsa.loop.fitting.loop_fit`):
- NUTS with a dense mass matrix;
- 4 chains of 1,000 warm-up and 1,000 draws, seed 0;
- one refit at `target_accept` 0.95 if the first fit fails the convergence gate;
- no time limit.

### 3.2 The predictions

**What is frozen:** each model's posterior-mean probability of each choice class, on every
display of the design pool (`src.rsa.loop.novelty.plain_pool`: 1,597 displays, pool digest
recorded in each file).

**The files:**
- the finals: `data/rsa/live/finals_pool_preds_final.json` (commit 8b34863c; all three
  converged);
- the bar: `data/rsa/live/bar_pool_preds.json` (`src.rsa.prereg.pool_fit`, same data file and
  settings).

**The model code:**
- the finals: `data/rsa/live/finals/` (sha256 prefixes `a80185358d52967b`, `d1d15e00bf421c58`,
  `0a83187b7398171a`, as in the predictions file);
- the starting models: `src/pipelines/outer_loop/projects/rsa_reference/seed_models/`;
- the promoted seeds: `data/rsa/live_seeds/models/`.

The run refuses any predictions files but the two whose sha256 are in §8.

## 4. Design

### 4.1 Displays

**G displays from the design pool, in two halves** (`src.rsa.prereg.design`, seed 2026).

**The EIG half: G/2 displays, chosen by the campaign's own design rule** (`src.rsa.design.run`,
unchanged):
- the models: the three finals as the models going in (prior mass 0.5), and the bar models
  sharing the other 0.5. As in every campaign design, a bar model that is the same file as one
  already in, or the same hypothesis on the pool (RMSE < 0.002), is merged into it. Here that
  leaves 12 distinct bar models (`prereg_design.json` `merged`):
  - the starting and promoted copies of `rsa_l1` and `rsa_l2` (the same files);
  - `rsa_l2_singleton_feat_color_familiar_l0` and `…_valence_l0` into
    `rsa_l2_singleton_distinct_feat_color_l0`;
  - `familiar_isolated_graded_costly_l2` into `isolated_graded_costly_l3`.

  The merge affects only how EIG divides its prior. The merged twins differ from the kept model
  by at most 0.0013 in probability on the design's displays (0.0061 for the costly pair). Each
  final's expected advantage over all three `l0` variants is the same to four decimals. The
  analysis (§6) scores all 17 bar models, merged or not;
- each model's posterior draws from the pooled fit;
- joint EIG, greedy;
- each display's responses as in the whole design.

**The random half: G/2 further displays drawn at random** from the rest of the pool, so the test
does not rest only on displays chosen to favour a disagreement. Each pick is uniform among the
displays that keep the quotas reachable.

**Quotas: both halves meet the campaign's minimum shares:**
- 15% two-object displays;
- 25% three-object displays;
- 20% mumble (prior) trials;
- 50% word trials.

### 4.2 Trials

**N trial lists:** each person answers 13 designed displays plus 2 catch trials, after a
practice trial (`trial_lists`, the campaign's balanced subsets). G is a multiple of 104 and N a
multiple of G/13, so every display gets the same number of lists.

**The page, the pay and the listing are the campaign's, unchanged:**
- "Who is Bob talking about?";
- $0.50 for a 3-minute listing;
- desktop only;
- approval rate ≥ 98%.

## 5. Participants

**Recruitment:** Prolific, one study of N places (`scripts/rsa/live/rsa_prereg.yaml`).

**Eligibility:** everyone who took part in any earlier RSA study is excluded (Prolific's
blocklist over every earlier RSA study, `exclude_earlier_participants_from: project`). That
includes the campaign's 1,800 people, so every participant is new.

**Exclusion:** a participant who misses any catch trial is excluded entirely (the campaign's
rule, `max_catch_errors: 0`). There are no other exclusions: no time-based exclusions and no
outlier removal.
- Prolific's held submissions are approved or rejected on Prolific's own grounds; that does not
  change which data are analysed.

**Stopping:** collection ends when Prolific reports N submissions, or after 48 hours.
- If it ends short, the study is paused and the collected data are analysed as they are; the
  shortfall is reported.
- No interim look at the outcome.
- Recruitment is not extended on the basis of the results.

## 6. Analysis

The analysis is `src.rsa.prereg.analysis`, run once on `experiment1/data/modelled.csv` (catch
trials removed). It was tested on simulated data before the freeze (`tests/test_rsa_prereg.py`).

### 6.1 Each model's score

1. **Per trial:** the log of the model's frozen probability of the chosen class on that display.
2. **Per model:** the sum over all modelled trials (lpd).

### 6.2 The primary test

1. **The bar:** the promoted seed with the highest lpd on these data. It is chosen after the
   fact, as in claim 2, so the bar is as strong as it can be and the test is conservative.
2. **The difference, per final f:** diff_f = lpd(f) − lpd(best promoted seed).
3. **Its standard error, clustered by display:** sqrt(G · var_g(Σ_{trials of display g} d_i))
   (`src.models.clustered_se.cluster_dse`).
4. **The test:** z_f = diff_f / SE_f, with one-sided p_f = 1 − Φ(z_f).
5. **The decision:** Holm at familywise α = 0.05. H1_f is supported when Holm rejects its null.

### 6.3 Reported beside it, not tested

- each final against the best starting model and against each of the 17 bar models;
- every model's lpd;
- the primary comparison within each half of the design (EIG, random);
- recruitment, exclusions and timing, as in the campaign reports.

### 6.4 If something goes wrong

- **A display missing from the pool, or a frozen probability of 0 for an observed choice:** the
  analysis raises; neither should occur, since every model has a lapse. Any such case and its
  handling is reported as a deviation.
- **Any other deviation from this plan:** reported with its reason, in a dated addendum and in
  `CAMPAIGN_LOG.md`.

## 7. Sample size

G and N come from `src.rsa.prereg.power`. It runs §6's analysis on simulated experiments: each
display gets m = ⌊N × (1 − 0.078) × 13 / G⌋ responses, using the campaign's 7.8% catch-exclusion
rate.

### 7.1 Two power calculations

**1. Simulation.** For each final f, responses are drawn from a truth λ·p_f + (1 − λ)·p_b, where
b is the promoted seed nearest f on the design. λ is set so that f's expected advantage over b is
δ nats per trial.
- **The sizing value:** δ = 0.0078. This is chain 2's experiment-3 committed model over the best
  promoted seed, the smallest of the three chains' experiment-3 advantages (0.0181, 0.0223,
  0.0078).
- **Also run, not used for sizing:** δ = 0.0058 (chain 2, experiment 2) and f as the truth.
- **The designs:** random designs within the quotas (conservative: the real design's EIG half
  aims at the disagreements), then the frozen design itself.

**2. Empirical scaling.** Claim 2's own tests, at 52 displays and about 47 responses a display,
grow as z ∝ √G at a fixed number of responses per display. With chain 2's experiment-3 test
(z = 1.27), the power at α = 0.05 (Holm's last step) and at α/3 (its first step) is:

| G | at α = 0.05 | at α/3 |
|---|---|---|
| 104 | 0.56 | 0.37 |
| 208 | 0.82 | 0.66 |
| 312 | 0.93 | 0.84 |
| 416 | 0.97 | 0.93 |

The scaling holds only at m ≥ 47 responses a display.

### 7.2 The rule

Take G from the grid {104, 208, 312}, and N as the smallest multiple of G/13 (so that the lists
balance), such that both:
- (a) every final's simulated power is ≥ 0.90 under δ = 0.0078, on random designs;
- (b) the empirical scaling gives ≥ 0.90 at α = 0.05, with m ≥ 47.

**G = 416 is not possible.** The pool has 54 two-object displays, and two disjoint halves of 208
under the 15% quota need 62. `src.rsa.prereg.design` refuses it before selecting.

### 7.3 Results (`data/rsa/prereg/power.json`, 2,000 simulated experiments per cell)

**(a) holds at every grid point from G = 104, N = 768.**
- The weakest final, chain 0's, has power 0.98–1.00 under δ = 0.0078; the other two have
  0.99–1.00.
- The simulation is optimistic. Its only noise is response sampling, so it lacks the display-
  to-display misfit that real data have (claim 2's SEs were several times larger).
- For chains 1 and 2, λ is capped at 1: even if the final were exactly right, its expected
  advantage over the nearest promoted seed is below δ. Averaged over the whole pool, it is 0.0069
  (chain 1, nearest `rsa_l2_singleton_feat_color_familiar_l0`) and 0.0053 nats a trial (chain 2,
  nearest `rsa_l2_singleton_feat_color_valence_l0`).
- On their 156 most diagnostic displays, it is 0.034 and 0.020. Those are the displays the EIG
  half aims at.

**(b) decides it:** G = 312 and m ≥ 47. That is N = 1,224 (51 cycles of the 24 lists), m = 47
at the campaign's 7.8% exclusion rate, about $816.

**Chosen (PI 2026-10-15): G = 312, N = 1,224.**

## 8. Frozen values (2026-10-15)

| item | value |
|---|---|
| code | `src/` and `scripts/` as at `507b5b9b`, the commit the design job (Sherlock 47967799) staged; the freeze commit changes only documents and the launch config. The registered bundle holds a copy of `src/rsa/prereg/` with hashes |
| G (displays), N (participants), m (responses a display) | 312 (156 EIG + 156 random), 1,224 (51 cycles of 24 lists; 51 lists per display), m = 47 at the campaign's 7.8% exclusion rate |
| `design_sha256` (`prereg_design.json`) | `bc7d60e5b84f0e09b1578da54e2ae73975b67e8f05f97a10cea75df6f8ceee02` |
| `trial_lists.json` sha256 | `a11b6cc6aeac1b70b8ba47174099bc1e264db722cea25ce44d17dfbb54a12672` |
| `finals_pool_preds_final.json` sha256 | `6a0a476f24e21166dd5af9d3f6538addd39db011b62d96a1adcd505a1d94f4bf` |
| `bar_pool_preds.json` sha256 | `704575376be1305a92506f7270c27c67524f26ffeb0391687d4178d5974de269` |
| pooled data sha256 (the bar fits' and the design's) | `19af0f544ed5c0e0cf902fdce55c259ad2fbaf697ccef6a21f9617375f114f23` |
| simulated power of the frozen design, per final (δ = 0.0078) | chain 0 0.995, chain 1 1.000, chain 2 1.000 (δ = 0.0058: 0.905, 0.997, 1.000; `design_power.json`) |
| quotas met, per half (EIG / random; minimum) | two-object 23 / 23 (23); three-object 39 / 39 (39); mumble 46 / 31 (31); word 110 / 125 (78) |
| design merges (EIG prior only; §4.1) | no final merged or screened out; 17 bar entries → 12 distinct |

## Addenda

(none)
