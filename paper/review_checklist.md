# Paper review checklist: `auto-psych-pnas`

Written 2026-10-30. The paper's claims, methods and numbers, checked against the code, run
records and figure data.

**Line references**
- "main" = `paper/auto-psych-pnas/auto-psych.tex`
- "SI" = `paper/auto-psych-pnas/supporting-info.tex`

---

## What checks out (recomputed or matched to the records)

- **Bias statistics (main:143):** recomputed from the 9 `responses.csv` files (n = 361 participant-experiments):
  - side: 0.503 [0.495, 0.511];
  - heads: 0.501 [0.493, 0.509];
  - alternation: 0.658 [0.639, 0.677].
- **Distinct pairs:** 416 is right.
- **Fig. 2 labels:** all match the figure data.
  - subjective randomness: +3,260 / +3,023 / +2,669;
  - reference games: +198 / +61 / +8;
  - number game: +1,592.
- **Fig. 3:** every r and RMSE value reproduces exactly. They are weighted by number of trials, which the caption could say.
- **Table S1:** every number matches `paper_table_without_run{1,2,3}.csv`.
- **Personal pattern-detection gain:** 17 population-level parameters and 6 person-level vectors. The most complex seed has 7.
- **Reference-games winner:** β = −0.31 [−0.40, −0.22] (`c_avoid`), and 17 parameters.
- **Recovery:**
  - All 17 held-out-seed cells beat the best other seed.
  - The alien cells end at RMSE 0.006–0.022, against 0.012–0.13 for the held-out seeds, so "even stronger recovery" holds.
- **SI EIG details (SI:67, 73):** these match the code the subjective-randomness runs used.
  - 64 pairs, N = 40;
  - 500/500/2 refits (replaced on 2026-10-10, after these runs);
  - 200 draws, 40 new participants;
  - the 2-MCSE stopping rule, rescoring the top 512 and the full set every 16 picks.
- **Other methods details:** these match the runs and code.
  - the 3-hour pause;
  - 5 rounds × 6 candidates;
  - 8 test statistics at p = .05.
- **Prompts:** `implement.txt`, `critic.txt` and `theorist.txt` match the prompts that actually ran.

---

## Wrong numbers or claims (results)

- [x] **main:143, "416 of 756":** should be **576** (9 × 64).
  - 416/576 = 72%, the figure the text reports.
  - 416/756 would be 55%.
- [x] **main:143, "median … between six and eight flips":** the median is **8 in all 9 experiments**. 453 of the 576 pairs are 8 flips long.
- [x] **main:143, "largely not being used across experiments or across runs":** misleading.
  - Experiment 1 had essentially the same design in every run: runs 2 and 3 share all 64 pairs, and run 1 shares 57.
  - The design seed was fixed, and experiment 1 uses the prior predictive.
  - The runs only diverge from experiment 2 onward
I reviewed this and think it's ok since experiments 2 and 3 mostly use different stimuli
- [ ] **Number game, Fig. 2 caption and main:163, "fit to the full dataset":** `data/paper/figure_data/number_game.json` contains **run 1 only**.
  - That is 7,560 responses (840 items × 9), not the 272,700-response bank.
  - Fig. 3's number-game panel is also run 1 only.
  - Runs 2–3 are not reported anywhere. This conflicts with main:130 ("three independent runs for each domain").
  - Separately, `scripts/paper/README.md` says the panel pools 22,680 responses from all three runs, which the data file doesn't match.
These are still waiting on new results from the cluster
- [ ] **Number-game seeds (main:161, Fig. 2 labels):**
  - The dataset README says the second Bigelow & Piantadosi seed (the interval-midpoint prior) is a variant "which no paper proposed", yet both are labeled "Bigelow & Piantadosi (2016)".
  - Both have since been replaced by the 2016 paper's Mixture and Independent models.
  - The figure says "Tenenbaum (2000)" while the text cites `tenenbaum1999bayesian`.
Daniel is re-running number game using the models from the actual paper and I'll report those when the results are in
- [x] **Fig. 2 caption colors:** the caption says "grey and green", but the figure colors discovered models by run (green, blue, pink). The best discovered models are the dark points.
Fixed
- [x] **Fig. 2 caption, reference games "fit to data from the preregistered follow-up":** the models were fit to the campaign's pooled data with predictions frozen in advance, then scored on the pre-registered data. That is what makes the test prospective.
- [x] **Reference-games seeds (main:151, "seeded … using the models from ProbLang V2"):** the live chains did not start from the textbook models.
  - They started from 12 *promoted* seeds: agent-written models from an earlier inner-loop run on the published data (Sherlock run 2), plus `rsa_l1`/`rsa_l2`.
  - These were split across the chains on purpose ("diversified starts, not replicates", `docs/auto_rsa/CAMPAIGN_LOG.md` #22), so "three replicates" is inaccurate.
  - The "best seed" in Figs. 2–3 is one of these agent-written models (`rsa_l2_singleton_distinct_feat_color_l0`).
- [x] **main:151, unfinished sentence ("found that the seeds."):** the pre-registered result is that 2 of 3 finals beat the best promoted seed (Holm-corrected).
  - chain 0: +197.9 ± 35.1, z = 5.64;
  - chain 2: +60.8 ± 22.9, z = 2.66;
  - chain 1: +8.1 ± 24.1, z = 0.34, not significant.

  All three beat the best textbook model (z = 8.0–10.3). The study recruited 1,225 people and kept 1,108 (14,404 trials). Also, "Figure … compared" should be "compares".
- [x] **main:155, "maximum of 4 for the seed models":** true only for the 5 textbook models. The promoted seeds have 2–7 parameters, and the comparison bar has 6.
- [x] **main:155, salience "whether it is in color" / "distinctive set of features":**
  - In the fit, `w_color` = −0.02 ± 0.98, which is its N(0, 1) prior; the live displays never varied color.
  - `w_distinct` = 0.015 [−0.006, 0.036].
  - Salience actually comes from being a singleton (0.73), feature count (0.30) and typicality (−0.49).
- [x] **main:147, winner description:**
  - The pooled posterior for the longest-run weight is negative: μ_δ = −0.87 [−1.59, −0.11]. So it is not a penalty on average, though the spread across people is large (σ_δ = 5.6).
  - The defining mechanism is missing: one personal pattern-detection gain that scales all the structural cues. The personal lapse rate is also missing.
  - "6 individual-level parameters" should say 6 person-level parameter *vectors*.
- [ ] **Ablations (main:180, Fig. 4):** several mismatches with `docs/sr_ablation.md` and `docs/auto_rsa/CONSOLIDATION.md`.
  - [ ] Only two domains were ablated. The number game was not (`docs/ng_ablation.md` is a plan), so "in each domain" is wrong.
  I will add number game
  - [x] Elimination removes *mechanisms*, not parameters: 15 for subjective randomness run 1, 13 for reference games. Person-level variation steps personal → shared → off.
  reworded
  - [x] Each round tries every *single*-mechanism removal, not "every variant that removed at least one parameter".
  - [x] The stopping rule is relative to the *current* model, not the full model: stop when the cheapest removal costs ≥ 2 clustered SEs of that comparison.
    - Subjective randomness stopped when removing antisymmetry cost 37.3 ± 15.9 (2.34 SE).
    - The lean model is 13.7 ± 13.6 behind the full one.
  - [x] The cross-validation is 5-fold grouped by stimulus pair. People are not held out.
  - [x] **The reference-games panel is not the winner.**
    - It is a superset of all three chains' finals: a plain RSA core plus 13 mechanisms, 16 parameters.
    - It leaves out some of the winner's terms (typicality, solitary foil, inhibition leak, valence/color), so "rewritten models make the same predictions as the original" is false here.
    - Its full fit did not converge (the hollow point).
    - The lean model (7 mechanisms, 9 parameters) was −81.1 ± 27.2 behind the winner on the pre-registered data.
    - `avoidable`, the winner's signature mechanism, was the first one dropped (1.87 SE).
    I will fix that
  - [ ] All three subjective-randomness finals were ablated, but only run 1's is shown.
    - run 2 kept 10 of 12 mechanisms;
    - run 3 kept 11 of 15.
    Yeah I know that
- [x] **Abstract:**
  - It still describes a single coin-flip testbed.
  - "The nested structure is critical" has no support in the current SI: the inner-loop ablation lived in `appendix.tex`, which is no longer `\input`. Under the current definitions it would also need a rerun (`results_update.md` M10).
  - "Three independent sequences of human experiments" fits neither the number game (no new experiments) nor the reference games (chains are not replicates).
- [ ] **Discussion (main:186), recovery "in three domains":** recovery is reported only for subjective randomness.
  - The SI recovery figure is `\includegraphics{}` with no file.
  - The reference-games and number-game text is placeholders.
  - Number-game recovery exists for 2 of 3 cells (`data/results/number_game_recovery-oct26/`); one cell ran out of memory.
---

## Methods errors

- [x] **main:228, pruning:** the live runs pruned at **4** stimulus-clustered SEs (recovery used 2), then kept at most 8 models. Delete the conflicting "within two standard deviations … added to the model registry" sentence.
- [x] **Trial order was not randomized (needs disclosing):** every participant saw the same 64-pair order.
  - Checked: 1 distinct order across the 40 participants of run 2, experiment 2.
  - The deployed page iterates `STIMULI.forEach`.
  - The SI's `implement.txt` is the no-shuffle version, which is the one that ran.
  - Only left/right placement was randomized.
- [ ] **Experiment-1 designs:** 9 joint-EIG picks plus 55 single-response fills, in all three runs. Experiments 2–3 were 64 joint-EIG picks each.
I will add more details on this in the SI
- [x] **main:214–228 and the SI describe only the subjective-randomness pipeline.**
  - [ ] Reference games:
    - grouped 5-fold CV instead of PSIS-LOO;
    - guarded selection;
    - pruning at 2 clustered SEs by display;
    - at most 8 *distinct* models carried;
    - stop after 2 rounds with no new best;
    - 200 people × 52 displays per experiment, with quotas and half the prior on the bar;
    - $0.50 per person;
    - agents with no network access.
  - [ ] Number game:
    - grouped 4-fold CV over example sets;
    - 280 items × 9 responses per experiment;
    - warm-start NUTS (200 tune / 500 draws);
    - no implementation agent.
  - [ ] SI:61 says Opus was used everywhere, but reference games used Gemini 3.8 Flash.
- [ ] **main:214:** "We begin by describing the subjective randomness domain" never happens.
- [ ] **main:224:** "two loops: an outer loop that …" names only the outer loop. Also, "We then selected" has the wrong tense.

---

## SI errors

- [ ] **SI:65:** "update the posterior π". Each experiment's design uses a uniform prior over the carried set; there is no posterior over models across experiments.
- [ ] **SI:158, MCMC settings:** recovery used 1000 tune / **1000** draws / 4 chains (per `holdout.json` `fit_kwargs`), not 2000 draws.
  - The softmax "posterior" is only reported; it is not used for selection.
  - The figure fits use other settings:
    - pooled subjective randomness: 4000 tune / 1500 draws, target_accept 0.95, no jitter;
    - number game: warm-start NUTS;
    - reference games: numpyro.
- [ ] **SI:160:** the critic writes the statistics; the *pipeline* simulates (1000 posterior-predictive replicates) and computes them.
- [ ] **Algorithm 1 is stale:**
  - It has an outer-loop `Theorist` step and a `Design()` agent; neither exists.
  - It shows single-stimulus EIG; the design uses joint EIG.
  - `M ← M ∪ {M_i}` should be the surviving set, with π ← Uniform. Also `M` should be `\mathcal{M}`.
- [ ] **Algorithm 2 is stale:**
  - It has a λ·length complexity penalty; the penalty is 0.
  - It selects by argmax π; selection is the ELPD-best model among those with reliable PSIS-LOO.
  - It has no pruning step and no cap.
  - "Valid and finite likelihood" should be the admission gates.
- [ ] **SI:170, participants:**
  - [ ] Pay was **$1.40** ($12/hr, 7-minute estimate), not $1.
  - [ ] The desktop-only filter is missing.
  - [ ] "Did not exclude participants from previous studies" is true only for run 1. The run 2–3 Prolific manifests have `previous_studies_blocklist`.
  - [ ] The 3.5-minute average can't be checked from the repository; the data have no timing columns.
- [ ] **SI:174–180, instructions are stale:**
  - "The two sequences may be different lengths." → "The two sequences will have the same length."
  - "You will complete 32 trials, which takes about 4 minutes." → "You will complete 64 trials, which takes about 7 minutes."
- [ ] **`figures/example-trial.png`:** shows a 7-vs-8-flip pair from the old design.
- [ ] **SI:217:** there were 17 recovery cells (5 for regularity detection, 4 for each other seed; 3 cells never ran), not five replicates each.
- [ ] **Table S1:**
  - [ ] The caption references `main-tab:mega-comparison`, which no longer exists.
  - [ ] It mentions "rank 0", but there is no rank column.
  - [ ] The tabular spec has 10 columns but only 9 headers.
  - [ ] SE(Δ) is the trial-level SE, while Fig. 2 uses stimulus-clustered SEs, which are 2–3× larger (e.g., 57.3 vs 128.3). Say which one is shown, or give both.

---

## Placeholders and typos

- [ ] **Placeholders:**
  - [ ] main:159 and main:161: TODOs (number-game design and seeds).
  - [ ] main:199: "NEW PARAGRAPH ON STIMULUS SELECTION".
  - [ ] main:7: CLASSIFICATION.
  - [ ] main:38–39: author contributions and competing interests.
  - [ ] main:40: corresponding author inside `\equalauthors`.
  - [ ] SI:185: reference-games instructions TODO.
  - [ ] SI:197: reference-games example figure TODO.
  - [ ] SI:208: empty `\includegraphics{}`.
  - [ ] SI:215: recovery placeholders for reference games and number game.
- [ ] **Typos:**
  - [ ] main:122 "Humans of make"
  - [ ] main:128 "out of starting set"
  - [x] main:147 "a sequences alternation rate", "the number heads"
  - [ ] main:155 "more likely be referring"
  - [ ] main:180 "the models that auto-psych achieved"
  - [ ] main:198 "maximially"
  - [x] main:220 "written a framework"
  - [ ] main:228 "in~\ref{SI-app:more-details}" is missing "SI Appendix"
  - [ ] SI:213 `{\sc auto-psych loop}`
  - [ ] SI:219 "easier discover"

---

## Possible further disclosures

These are from the earlier audit in `paper/results_update.md`; I did not re-check them, except where noted.

- [ ] A PSIS NaN bug dropped models during the live runs; run 2's experiment 2 started with 2 of its 8 models. Fixed in `943cdcc6`, after the runs.
- [ ] Run 1's experiment 1 was re-pruned at 4 SEs after first pruning at 2.
- [ ] Run 3's experiment 3 has 41 participants (re-checked).
- [ ] The subjective-randomness agents had network access; the reference-games agents had none.
- [ ] One participant's parameters in the pooled fit of run 2's winner did not converge, and that fit was accepted anyway (`accepted_fit_problems`). The population-level parameters converged.
- [ ] The IRB number (#20009) appears nowhere in the repository. Confirm it.
