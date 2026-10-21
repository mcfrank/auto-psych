# Paper methods: changes needed to match the current pipeline

**Status as of 2026-10-02**
- Re-checked against the combined paper in `auto-psych-combined/` and the code
  at `origin/main` `bc547ac`.
- No change to the methods in the code since the first version of this list:
  the two newer commits touch only docs and the live job's memory request.
- Both documents build cleanly from scratch with `latexmk` (main 7 pages,
  SI 15).

**Line references**
- "Main" = `auto-psych-combined/auto-psych.tex`.
- "SI" = `auto-psych-combined/supporting-info.tex`.
- The SI is now the only copy of the appendix: `appendix.tex` is no longer
  `\input`.

## Open decision that affects the methods

- [ ] **Agent backend for recovery.** Main 212 and SI 61 now both say Claude
  Code + Opus 5.5 for everything.
  - The recovery sweeps run OpenCode + `google/gemini-3.1-pro-preview`
    (`scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml`,
    `agent:` block).
  - The live runs use Claude Code + `claude-opus-5-5` (`full_run_sr-oct26.yaml`).
  - Either state both backends, or rerun recovery with Opus.

---

## Problems introduced by the latest edits

- [ ] **Main 111: the seed descriptions contradict the new last sentence.**
  - The four descriptions are still the old, deleted seeds: encoding
    compressibility, Bayesian diagnosticity, window typicality and prototype
    similarity.
  - The new sentence calls them "designed to be faithful to the accounts
    developed in the original papers".
  - Replace the descriptions with the current seeds (table under Main 111
    below).
- [ ] **Main 228: two conflicting pruning sentences.**
  - "All models whose ELPD is 2 dse worse than the best model are removed" is
    right only for recovery. The live runs prune at 4 dse, and the dse is
    clustered by stimulus.
  - The cap of 8 models is missing.
  - The old sentence that follows it ("All models that are within 2 dse of the
    incumbent … added to the model registry") should be deleted.
- [ ] **Main 228:** "can be found in~\ref{app:more-details}" is missing the
  word "Appendix" before the reference.
- [ ] **Main 219:** "heirarchical" → "hierarchical".
- [ ] **SI 67:** the placeholder "ADD DETAILS OF EIG." is still in the text.
- [ ] **SI 67: a sentence in the wrong section.** "The agent is prompted to
  ensure that each model represents a single, distinct hypothesis…" sits in
  the outer-loop section, where no agent proposes models. Move it to the
  theorist paragraph (SI 164).
- [ ] **Main 40: corresponding-author text in the wrong command.**
  - The corresponding-author text is inside `\equalauthors`, which renders it as
    an equal-contribution footnote.
  - `\author[a,1]` marks the first author with that footnote.
  - Use `\correspondingauthor`, and drop the `1` unless authors contributed
    equally.

---

## Main text

### Main 93 — intro overview of the two loops
- [x] The outer loop no longer "proposes probabilistic cognitive models".

### Main 103 — Figure 1 caption
- [x] Stimuli: 64 chosen from all same-length pairs at lengths 2–8 by EIG.
- [ ] "The best-fitting model is carried forward to the model registry" → the
  surviving models (up to 8) are carried.
- [ ] Delete "In practice, we have an additional theorist agent in the outer
  loop…".
- [ ] Optional: say it is the **joint** EIG of the set.

### Figure 1 PDF (`figures/autopsych-methods.pdf`)
- [ ] "add best-fitting model to registry" → carry the surviving models.
- [ ] The inner-loop box "choose & score stimuli" should be "fit & score".
  Stimulus choice happens in the outer loop.
- [ ] Optional: show the admission gates, the pruning step and the ledger.

### Main 109 — paradigm
- [x] Pairs of same-length sequences.

### Main 111 — seed models
- [ ] Replace the four descriptions (see "Problems introduced" above).

  | Current seed | Suggested paper name | What it computes | Free parameters |
  |---|---|---|---|
  | `falk_konold_dp` | Encoding difficulty | Falk & Konold (1997) Difficulty Predictor: runs + 2 × alternating runs under the minimising parse, not length-normalised | `beta`, `side_bias` |
  | `motif_stack` | Regularity inference | Griffiths et al. (2018) four-motif stack automaton (mirror, complement, duplication); randomness = log P(fair) − log P(regular) | 7 (`delta`, `alpha`, `repetition_weight`, `mirror_share`, `complement_share`, `beta`, `side_bias`) |
  | `finite_experience_occurrence` | Finite experience | Hahn & Warren (2009): log P(string occurs at least once in 20 fair flips) | `beta`, `side_bias` |
  | `local_representativeness` | Local representativeness | Kahneman & Tversky (1972) operationalisation: local balance at window sizes 2–4 + an over-alternating prototype + a periodic-template penalty | 5 (`theta_alt`, `alt_weight`, `periodic_share`, `beta`, `side_bias`) |

- [ ] "Faithful" needs a qualifier for `local_representativeness`: its own
  manifest calls it "theory-inspired", because K&T published no quantitative
  equation.
- [ ] Optional: `motif_stack` and `finite_experience_occurrence` refuse
  cross-length pairs, which is one reason the design uses same-length pairs
  only.

### Main 123 / 130 / 132 — recovery evaluation (update with the new results)
- [ ] **130:** "all possible pairs of stimuli of length up to eight" → every
  distinct same-length pair at lengths 1–8, minus every pair used in any
  experiment's training data.
- [ ] **123 and 132:** "the best seed model … fit to all of the data" → the
  ELPD-best other seed, refit on each experiment's cumulative data.
- [ ] **123:** each experiment shows a seed step plus up to 5 rounds.

### Main 212 — agents
- [x] Three agents: theorist, critic and implementation. The design has no
  agent.
- [ ] Backend: see the open decision above.

### Main 218–220 — theory framework
- [x] Inputs: a sequence pair, an anonymized participant index and a trial
  index; models can compute their own features.
- [x] Models can be hierarchical, with person-level parameters.
- [ ] "typically includes a bias parameter" → **always**. Every seed and every
  alien model has `beta` (Uniform 0.2–12) and `side_bias` (Uniform −2 to 2).
- [ ] Optional: say that models see **only** the raw sequences and choices; no
  features are precomputed.

### Main 224 — stimulus design
- [x] All pairs of same-length sequences at lengths 2–8; uniform prior over the
  carried models.
- [ ] State that **64** stimuli are selected per experiment.
- [ ] The EIG is the **joint** EIG of the stimulus set. Each stimulus yields a
  count k ~ Binomial(N, p_left), with N = 40 participants.
- [ ] Experiment 1 uses prior-predictive draws. Experiments ≥ 2 refit every model
  to all data so far and use posterior-predictive draws.

### Main 226 — implementation and deployment
- [x] The pipeline deploys, recruits and polls.
- [x] An unfilled study is paused after 3 h.
- [ ] IRB #20009 is not recorded anywhere in the code. Confirm it.

### Main 228 — inner loop
- [x] Critic first, 8 statistics, 5 rounds × 6 candidates, hypothesis search.
- [x] Slot roles (3 explore / 2 refine the incumbent / 1 refine a chosen model).
- [x] Admission gates (finite log-likelihood, convergence, novelty).
- [ ] Pruning: see "Problems introduced". The rule is:
  - pruned when ELPD diff > k × dse, against the best trusted model;
  - the dse is clustered by stimulus (`src/models/clustered_se.py`);
  - k = 4 live (`prune_dse_multiplier: 4`), 2 in recovery;
  - then at most 8 models are kept;
  - the survivors are carried.
- [ ] Optional: the remaining gates — a 30-min fit limit and the data contract.
  Novelty is RMSE ≥ 0.002 on a 512-pair pool.

### Main 197 — limitations
- [ ] The placeholder ("NEW PARAGRAPH ON STIMULUS SELECTION…") is unfinished.
  The design's `n_random` knob (`design.n_random` in the recovery config) can
  run random or mixed designs if you want an ablation that addresses Dubova et
  al.
- [ ] Main 192: "participant averages rather than individual variation" is
  partly addressed now that the leading live models are person-level.

---

## Supporting information

### SI 61 — harness and model
- [ ] See the open decision above.

### SI 63–75 — outer loop
- [x] The outer-loop theorist paragraph is removed.
- [ ] **65:** "update the posterior π" → the design prior is uniform over the
  carried models; there is no posterior over models across experiments.
- [ ] **67:** the placeholder, and the misplaced hypothesis sentence (see
  "Problems introduced").
- [ ] **69–73:** replace the single-response equation with the joint EIG of a
  stimulus set:
  - S = the set of chosen stimuli;
  - the outcome at stimulus s is k_s ~ Binomial(N, p_left(s));
  - EIG(S) = H(M) − E_k[H(M | k, S)].
- [ ] **69–73:** describe the selection (`src/models/eig_selection.py`,
  `src/pipelines/outer_loop/eig.py`):
  - **Greedy:** each step adds the pair with the largest joint-EIG gain.
  - **Stop rule:** stop when the best gain is within 2 Monte Carlo standard
    errors of zero.
  - **Fill:** remaining slots get single-response EIG, conditioned on the picks
    so far.
  - **Leave-one-out:** each Monte Carlo scenario's likelihood average leaves out
    the draw that generated it.
  - **Lazy search:** float32; the whole pool is rescored every 16 picks, and
    otherwise only the top-ranked candidates in batches of 512. Validated
    against exact float64 greedy (`validate_lazy_eig.py`).
  - **Person-level models:** a model with a participant effect is predicted as
    40 new participants, so its per-person parameters are drawn from the
    fitted population distribution.
- [ ] **75:** "the agent deploys … recruits … polls" → the pipeline does this
  (as Main 226 now says).

### Algorithm 1 (SI 77–104)
- [ ] Remove the `if n > 1: M ← M ∪ Theorist(problem, M)` block.
- [ ] Replace `C ← Design()` with C = all same-length pairs at lengths 2–8.
- [ ] Replace per-candidate EIG with greedy joint-set EIG. Prior predictive when
  n = 1, posterior predictive given D when n > 1.
- [ ] Replace `M ← M ∪ {M_i}` with M ← the surviving set, and π ← Uniform(M).

### Algorithm 2 (SI 106–152)
- [ ] `Score`: drop `λ · length(m)`; the complexity prior is 0
  (`DEFAULT_COMPLEXITY_PRIOR_CONST` in `src/pipelines/inner_loop/scoring.py`).
  The softmax posterior is only reported.
- [ ] `m† ← argmax π` → m† = the best model by ELPD-LOO rank among models with
  reliable PSIS-LOO (`_best_exportable_model`).
- [ ] Critic: J = 8 statistics, B = 1000 replicates, plus a BH-FDR q. One retry if
  the critic writes no usable statistic, otherwise no critique that round.
- [ ] Candidate loop:
  - slot roles;
  - inputs: the ranked models (ELPD diff ± dse), the ledger or refinement menu,
    and the critiques;
  - candidates are fitted in parallel, then admitted one by one.
- [ ] Replace "valid and has finite likelihood" with the admission gates.
- [ ] Add retry/repair: a slot that wrote no model is re-spawned once; a
  rejected candidate is re-spawned once with the rejection reason.
- [ ] After the R rounds: prune at k × clustered dse, cap at 8, carry the
  survivors.
- [ ] Remove `λ` from the parameter list. Add the novelty threshold, the pruning
  k and the cap.

### SI 155–164 — inner loop
- [x] Two agent types, theorist and critic.
- [x] Hypothesis search is described in the theorist paragraph.
- [ ] **159:** recovery MCMC is **1000 warm-up, 1000 draws, 4 chains**, not
  1000/2000/4.
  - Both conditions use target_accept 0.8; `motif_stack` declares a 0.9 floor.
  - A near-miss fit (≤ 2% divergent, R-hat ≤ 1.2, bulk ESS ≥ 20) is refit once at
    0.95.
  - The human settings (2000 / 3000 / 4) are correct.
- [ ] **159:** delete the 0.05-per-line complexity penalty (removed 2026-09-29)
  and "Scores are fed into a softmax … posterior π" (it is only reported now).
- [ ] **159:** add the LOO-reliability rule (`src/models/loo_reliability.py`):
  - a tolerated share of high-Pareto-k trials;
  - trials whose log-likelihood is constant are exempt as exact;
  - not ArviZ's blanket any-k > 0.7 warning.
- [ ] **161–162 (critic):** the critic agent only writes the statistics
  (`test_statistic(df) -> float`). The pipeline runs the check
  (`src/critique/ppc.py`):
  - 1000 posterior-predictive replicates;
  - a two-sided empirical p;
  - a Benjamini–Hochberg q.

  Candidates see the statistics with p ≤ .05, each with its q.
- [ ] **164:** "discouraged from stitching together…" applies only to
  **exploratory** slots. Refinement slots may borrow one component from another
  model, as a single stated change.
- [ ] **164:** add the slot roles here in more detail than Main 228:
  - the 11 exploration lenses;
  - the incumbent named in the brief;
  - the refinement menu of live and pruned models.
- [ ] **164:** add the **hypothesis ledger**:
  - `attempted_hypotheses.jsonl` records every admitted, rejected and pruned
    hypothesis;
  - it is carried across experiments;
  - exploratory slots see it as "already tried — do not re-propose".

### SI 166–191 — human experiments
- [ ] **171:** pay is $12/hr for an estimated 7 min ($1.40), not "$1". Replace
  the 3.5-min average with the new runs' average.
- [ ] **171:** add:
  - desktop only;
  - each pair's sides swapped with probability 0.5;
  - a participant returning in a later experiment of the same run keeps their
    earlier ID.
- [ ] **177–181 (instructions):**
  - "The two sequences may be different lengths." → "The two sequences will have
    the same length."
  - "You will complete 32 trials, which takes about 4 minutes." → "You will
    complete 64 trials, which takes about 7 minutes."
  - Source: `src/pipelines/outer_loop/projects/subjective_randomness/problem_definition.md:49-57`.
- [ ] Replace `figures/example-trial.png` with a screenshot from the new
  experiments.

### SI 193–204 — inner-loop ablation
- [ ] **202:** the caption's baseline wording, as for Fig. 2.
- [ ] Describe the ablation: `--inner-loop-iterations 0` removes the critic and
  all candidate agents. The EIG design, end-of-experiment pruning and the cap
  still run.
- [ ] A ready-made wrapper exists only for the alien condition
  (`run_impossible_no_inner_loop_test_retest.sh`). For the literature ground
  truths, use `INNER_LOOP_ITERATIONS=0 submit_holdout_test_retest.sh`.

### SI 239–252 — prompts
- [x] The outer-theorist and design prompts are removed, and the merge-conflict
  note is gone.
- [x] `prompts/implement.txt`, `critic.txt` and `theorist.txt` are identical to
  the current `3_implement.md`, `critique.md` and `pymc_theory.md`.
- [ ] Say that `theorist.txt` is only the first piece of the candidate's prompt.
  `_build_candidate_prompt` (`src/pipelines/inner_loop/candidate_agent.py`)
  appends, in order:
  - file-writing instructions;
  - any retry or repair note;
  - `CONTEXT.md`;
  - the role-specific brief;
  - `existing_hypotheses.md`;
  - `attempted_hypotheses.md` or `refinement_menu.md`;
  - `critiques.md`.

  Describe this, or show one assembled example.
- [ ] Optional: say that the critique context is inlined into the critic's
  prompt (`critique_round._build_critique_prompt`).

### Template leftovers
- [ ] **SI 13–15:** title, authors and corresponding author are placeholders.
- [ ] **SI 48:** remove `\instructionspage`. It causes the only two undefined
  citations in the build.
- [ ] **Main 35–39:** significance statement, author contributions and competing
  interests.
- [ ] Optional: in the main text, "Appendix~\ref{…}" → "SI Appendix,
  section~\ref{…}" now that the SI is a separate document.

---

## New methods not described anywhere yet

- [ ] **Agent isolation** (`src/runtime/agent_sandbox.py`). This is the main
  argument that recovery doesn't leak the answer.
  - Agents run in bubblewrap. Their file tree is read-only except their own
    directory, and contains only `src/` and the run tree: no docs, tests or
    literature.
  - Their environment is allowlisted.
  - Before agents start, a scan stops a recovery cell whose files name its
    held-out ground truth (`scan_gt_name.sh`).
- [ ] **Hypothesis ledger** (see SI 164).
- [ ] **LOO-reliability rule** (see SI 159).
- [ ] **Recovery setup:**
  - 5 replicates per ground truth;
  - alien models generated with `beta` = 4, `side_bias` = 0;
  - literature ground truths at each family's default parameters;
  - the evaluation pool and baseline as under Main 123/130.
- [ ] Optional: the new metrics, if you report them:
  - incumbent changes and discovered models (`src/subjective_randomness/incumbent.py`);
  - test-retest reliability and ICC (`holdout_test_retest.py`);
  - model-averaged trajectories;
  - Pearson r, KL regret and calibration besides RMSE.

---

## Not methods, for reference once the new results arrive

- Everything quantitative in Results comes from the June runs:
  - Fig. 2, Fig. 3 and the ablation figure;
  - Table 1 and SI Table S1;
  - Main 145–178 and the claims in the Discussion (182–186).
- Counts that change under the new design: 9 experiments × 64 = **576** pairs,
  not 288; 9 × 40 × 64 = **23,040** trials, not ~11k.
- Main 234 (data availability): release only `responses.csv`, whose participant
  IDs are numeric, never `raw_collected/`, which has Prolific IDs.
