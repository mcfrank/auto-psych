# Run B (the rehearsal): result

From the local Sherlock session, 2026-10-25, for the driver session. Handoff: `HANDOFF_rehearsal.md`
§3 and "Run B: let it finish". Earlier status: `RUN_B_STATUS.md`.

**Run B completed all three experiments**, in 2 days 13 h, for **$125.25** of recorded API spend.

What worked:
- **Every stage ran end to end.** Each design was frozen with a passing pilot.
- **Each experiment used a new response format:** a bet, then a rating, then a forced choice.
- **The leading model changed twice.**
- **`gt_name_mentions.txt` is empty:** no agent file names the truth.

What fell short:
- **The loop discovered almost nothing after experiment 1.** Of 84 candidate attempts, 65 (77%)
  were rejected as too slow, and experiment 3 admitted none.
- **Each change of leader after experiment 1 came from new data,** re-ranking models already
  carried, not from a new theory.

## 1. Brought back (`data/rsa_open/rehearsal_b/`, on `auto-psych-open`)

Included:
- `outputs/`: all three experiments;
- the fits' `.fit_cache/*.json` sidecars, force-added because `.fit_cache/` is gitignored;
- `ground_truth.json`, `gt_name_mentions.txt`, `DONE`, `code_history`;
- both jobs' logs as `job_<id>_log.txt`;
- the chain's `token_usage.jsonl` and `token_usage_summary.json`.

The partial copy of experiments 1 and 2 (1b93be1f) is replaced by this one.

Left out:
- **The fit caches, and the agents' `scratch/`, `.xdg_data/`, `.home/` and `.agent_shell/`.**
- **`ground_truth.py`.**
- **Every copy of E's trials:** `corpus.json`, `prior_corpus.json`, `loop/data/{E.jsonl,
  check_corpus.json}`, `loop/.cv/`. Unlike run A's pragmods-only corpora, run B's corpus is all of
  E, run 2's `all_trials.csv`, which includes sources whose trial-level data may not be committed
  (`DATASETS.md`). The simulated experiment data are kept (`data/block.json`, `rows.jsonl`,
  `loop/data/exp<n>.jsonl`).

Experiment 3's `lists.json` and `page_preview.html` are 51 MB each: over GitHub's 50 MB warning,
under its 100 MB limit.

## 2. Jobs and timeline

| job | code | outcome |
|---|---|---|
| 49001912 | e8b6acb8 | cancelled at 1 h 54 min for the faster-fitting redeploy (truth and design fits kept) |
| **49021326_0** | **99cb1e28** | **COMPLETED, Thu 09:49 → Sat 22:59 (2 d 13 h 10 min), 24 cores** |

| | design fits | designer and checks | loop |
|---|---|---|---|
| experiment 1 | (reused) → Thu 09:50 | 09:54 → frozen 12:09 (1 cycle) | 12:10 → Fri 15:29 (27.3 h, of which 9.4 h was §5's stall); 4 rounds |
| experiment 2 | Fri 15:29 → 16:36 | 16:40 → frozen 19:07 (1 cycle) | 19:07 → Sat 06:15 (11.1 h); 2 rounds |
| experiment 3 | Sat 06:15 → 08:15 (the 2 h limit) | cycle 1 08:19–10:19 **failed**; cycle 2 10:24 → submitted 11:31 → frozen 12:11 | 12:12 → 22:58 (10.8 h); 2 rounds |

Every loop stopped early by the stale rule: two rounds in a row left the incumbent unchanged.
With 5 rounds allowed, they ran 4, 2 and 2.

## 3. The designs

| | paradigm | response | size | EIG | pilot |
|---|---|---|---|---|---|
| 1 | `graded_belief_allocation` | 10-point bet | N 120 × 15, 3.9 min | 3.50 bits, 0.0337 bits/$ (floor 0.0120) | comprehension, catch, readable all 1.0; 4.2 min estimated |
| 2 | `probed_referent_likelihood_rating` | 7-point rating of one probed referent | N 200 × 20, 3.0 min | 3.87 bits, 0.0291 (floor 0.0129) | all 1.0; 3.1 min |
| 3 | `multi_competitor_referent_choice` | forced choice, 2–6 objects | N 200 × 28, 3.0 min | 3.58 bits, 0.0268 (floor 0.0122) | all 1.0; 3.4 min |

- **The designer changed the response format each time.** Each rationale argues the next format
  tests whether theories fitted to the earlier formats generalize.
- **Experiment 3's first cycle ran out its 120 min without submitting:** "the designer's session
  reached its time limit (120 min) without a submission". It drafted two paradigms, a probability
  slider and a speaker acceptability judgment, and validated and scored both, but submitted neither.
- **Cycle 2 explored choice, production and bet versions** and submitted a choice paradigm that
  passed every check. One more failure would have stopped the chain (`DesignerFailed`).
- **Screened out of experiment 3's design:** `familiar_isolated_graded_costly_l2` and
  `softmax_surprisal_isolated_costly_l4`, both "the fit did not finish within 120 minutes". One
  more design fit was used unconverged (min bulk ESS 77). Experiments 1 and 2 screened out none.
- **`record.json` has `review_required: true` for experiment 1 only.** It is `false` for 2 and 3.

## 4. The loop

### Candidates

Each experiment's ledger continues the previous one's. These are each experiment's own lines.

| | rounds | attempts | admitted | too slow (30 min) | other |
|---|---|---|---|---|---|
| experiment 1 | 4 | 38 | 9 | 24 | 2 no file, 3 not novel |
| experiment 2 | 2 | 23 | 2 | 19 | 2 hypotheses copied word for word from the refined theory |
| experiment 3 | 2 | 23 | **0** | **22** | 1 no file |
| **total** | 8 | **84** | **11** | **65 (77%)** | 8 |

**The 30-minute admission limit on all of E is the run's main finding.**
- In run B, rejections for being too slow went from 63% of attempts in experiment 1 to 96% in
  experiment 3.
- **Repairs rarely rescued a candidate.** By my matching of ledger contexts, 34 repairs followed
  a too-slow rejection: 32 were too slow again, and 2 were admitted. Those two,
  `rsa_l1_singleton_feat_color_valence_l0` and `feature_uncertainty_listener_2`, both later
  became the incumbent. **Correction:** `RUN_B_STATUS.md` said every repair of a too-slow
  candidate was too slow again. That was wrong.
- Run B′'s corpus-relative limit (83fb8fe4) is aimed at exactly this.

### Incumbents

| step | experiment 1 | experiment 2 | experiment 3 |
|---|---|---|---|
| start (carried set, new data) | rsa_l2_singleton_feat_color_valence_l0 | **rsa_l1_singleton_feat_color_valence_l0** | **feature_uncertainty_listener_2** |
| round 1 | rsa_l2_singleton_feat_color_valence_l0 | same | same |
| round 2 | **bounded_depth_listener** (discovered) | same | same |
| rounds 3–4 | bounded_depth_listener | — | — |

- **In experiments 2 and 3 the leader changed at the start step.** Adding the new experiment's data
  re-ranked the carried models before any theorist ran, and no round changed it.
- **All three leaders are discovered models.** Two of them, `rsa_l1_singleton_feat_color_valence_l0`
  and `feature_uncertainty_listener_2`, were admitted as repairs of too-slow candidates.

### Final standing (experiment 3)

"Behind" is in nats ± the clustered SE (grouped CV, except E).

| theory | E | exp1 | exp2 | exp3 |
|---|---|---|---|---|
| **feature_uncertainty_listener_2** (incumbent) | 139.5 ± 56.5 | 36.5 ± 18.7 | 44.4 ± 42.6 | **best** |
| rsa_l2_singleton_feat_color_valence_l0 | **best** | 27.9 ± 20.2 | 10.0 ± 59.0 | 2.5 ± 15.3 |
| rsa_l1_singleton_feat_color_valence_l0 | 45.0 ± 24.2 | **best** | **best** | 9.0 ± 13.2 |
| bounded_depth_feature_parsimony | 156.9 ± 57.5 | 10.7 ± 9.8 | 13.7 ± 22.1 | 9.2 ± 9.6 |
| bounded_depth_listener | 166.2 ± 58.5 | 8.7 ± 9.2 | 2.9 ± 18.5 | 11.6 ± 11.2 |

All five are eligible and trusted, and all five are carried. On exp3 every model is within about
1 SE of the incumbent, so experiment 3's data do not separate them.
- **The incumbent ranks fourth on E:** 139.5 ± 56.5 behind, 2.5 SE, inside the 4-SE guard.
- **On exp1 and exp2, `rsa_l1_singleton_feat_color_valence_l0` leads**, and the incumbent is behind.
- **The run does not score recovery of the truth.** Whether any of these is close to the truth
  (`crowd_discrim_confusion_chromatic_l2`, a model the chain did not start with) is open.

### Critic

It ran in every round (8 sessions): **8 of 8 statistics significant after FDR every time**, as in
run A.

## 5. The stalled fold refit (`RUN_B_STATUS.md` §3)

- **The stall:** one CV fold of `feature_parsimony_listener` took 5 h 50 min and failed the gate.
  Its unconditional, unlimited refit then took 5 h 14 min and failed again (R-hat 1.53, ESS 7).
- **The cost:** round 3 waited 9.4 h, and the model ended untrusted.
- **The fix:** run B′'s code refits near misses only and bounds each fold at 3× its full fit.

## 6. Spend (`token_usage_summary.json`: 1,232 calls, 0 missing usage, 0 usage-limit waits)

| source | calls | cost |
|---|---|---|
| theorists (gemini-3.8-flash) | 92 | $101.57 |
| designer (gemini-4-argon) | 4 | $15.26 |
| critic (gemini-3.8-flash) | 8 | $8.42 |
| pilot (gemini-3.8-flash) | 1,128 | not priced; 3.4 M tokens |
| **total** | | **$125.25** |

That was within the handoff's estimate of $150–250 plus the pilot, mainly because the loops
stopped early. The cancelled first job's 13 min of designer time is not in this log.

## 7. Next

Run B′ (`HANDOFF_rehearsal.md`, "Run B′") waits for the PI's go-ahead. Run B has finished, so the
node is free. B′ replays experiment 1's loop on the fixed code, in a fresh root
(`rsa_open_replay_b1`), with this run's ground truth copied in. **The comparison numbers for
experiment 1** are in §4: 38 attempts, 9 admitted, 24 too slow; incumbent `bounded_depth_listener`
from round 2; 4 rounds.
