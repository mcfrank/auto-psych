# Run A (the dry run): status and one blocking fit

From the local Sherlock session, 2026-10-21 16:00 PDT, for the driver session. Handoff:
`HANDOFF_rehearsal.md` §2. **No agent has run yet, so no API money has been spent.**

## 1. The attempts so far

| job | work root | code | outcome |
|---|---|---|---|
| 48810660 | `rsa_open_dry` | 74d36595 | failed in 4 min: the shared venv `$GROUP_HOME/venvs/auto-psych` did not exist |
| 48811977 | — | — | built that venv (`build_shared_venv`) plus google-genai 2.8.0, pydantic 2.13.4 and resvg-py 0.5.0; every import and a resvg render passed |
| 48812366 | `rsa_open_dry` | 74d36595 | cancelled at 64 min, before any agent ran: switched to 539a29e2 (your recommendation) |
| **48820332_0** | **`rsa_open_dry2`** | **539a29e2** | **running since 09:27 (6 h 30 min)** |

The current run is set up as follows:
- **The ground truth** is `rsa_open_dry`'s, copied into `rsa_open_dry2/private/ground_truth/`. The
  truth code and the seeds are unchanged between the two commits, so it was not refitted.
- **`env_check`** passed: "13 packages and 11 harness modules import; resvg renders".
- **The agent-tree checks** passed, and the expected variables were withheld from the agents.

## 2. Where it is: the design's context fits (`Chain.design_context`, experiment 1)

All 15 bar theories started fitting together at 09:31, each in its own pinned child process. The
concurrency works. Their cache files appeared in `experiment1/.fit_cache/` as each finished:

| finished | theory | | finished | theory |
|---|---|---|---|---|
| 09:34 | `literal_listener` | | 10:24 | `rsa_l2` |
| 09:47 | `rsa_l1` | | 11:17 | `rsa_l2_singleton_feat_color_valence_l0` |
| 09:52 | `lexical_preemption_heuristic` | | 11:33 | `familiar_isolated_graded_costly_l2` |
| 09:55 | `rsa_l1_salience` | | 11:34 | `rsa_l2_singleton_feat_color_familiar_l0` |
| 10:00 | `competitor_confusion_speaker` | | 11:45 | `isolated_graded_costly_l3` |
| 10:15 | `evaluative_prominence_l2` | | 12:56 | `rsa_l2_singleton_distinct_feat_color_l0` |
| 10:19 | `oddity_heuristic_listener` | | 13:02 | `softmax_surprisal_isolated_costly_l4` |
| **still running** | **`rsa_l1_shared_prior`** | | | |

**`rsa_l1_shared_prior` has been fitting for over 6 h 15 min, on one core.** The other 14 took
3–211 min.
- Its sibling `rsa_l1_salience` finished in 24 min. It has the same three parameters (`alpha`,
  `w_features`, `w_familiar`) and the same priors; the only difference is that `rsa_l1_shared_prior`'s
  `L0` also uses the salience prior.
- Nothing shows whether this is the first fit or the one refit at `target_accept` 0.95.

**Why it blocks:** the design can't start until every context fit returns. `design_context` calls
`loop_fit(..., time_limit_sec=math.inf)`, so one pathological fit stalls the chain, without limit,
until the job's own limit:
- the dry run's limit is 24 h;
- run B's is 5 days.

The design is then up to about 6 h and the two rounds a few more, so run A will probably need a
resubmit (it resumes, and the 14 fits are cached).

**This will recur.** Every experiment's design refits the bar on the data so far. From experiment 2
on, it reads the previous loop's cache, which holds a bar theory only if the loop fitted it on the
same corpus. So in run B the same fit can stall each experiment.

## 3. Options (yours)

1. **A time limit on the design's context fits, with a recorded rule.** For example, a theory whose
   fit is not back in N hours is left out of that design, and the omission is written down, as the
   closed design's `screened_out` was. A theory is never silently dropped.
2. **Diagnose the fit:** run `rsa_l1_shared_prior` alone on `prior_corpus.json` (here, or from the
   cloud on a sample) and look at its divergences, tree depth and step size. The port (rule (a))
   may give it a posterior the sampler handles badly, even though its sibling fits quickly.
3. **Let run A wait it out.** It costs compute only, and I resubmit at the 24 h limit if needed.

Until you say otherwise, I am leaving 48820332_0 running (option 3). If you push a fix, I'll
cancel, pull, and restart in `rsa_open_dry2` with a `RESTAGE_REASON`. The 14 finished fits are
reused if their cache keys don't change.

Logs: `$SCRATCH/auto-psych/rsa_open_dry2/logs/rsa_open_rehearsal_48820332_0.out`. Run dir:
`agent_trees/544a76a2b5f3d6aa/repo/_runs/chain_0/`.
