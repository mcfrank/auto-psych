# RSA loop smoke test 2: Gemini 3.8 Flash, all five sources, two rounds

Run 2026-10-07 01:05 to 08:00 UTC in a disposable claude.ai cloud container
(4 CPUs, 16 GB RAM, a 14.3 GB memory cgroup), on branch `auto-rsa-smoke2`
(from `auto-rsa` at `93fa1f5`), following `HANDOFF_smoke_test_2.md`. Agents:
opencode 1.18.35, `google/gemini-3.8-flash`, 2,400 s per agent. Fits: NUTS,
2 chains × (500 warmup + 500 samples).

Small outputs are committed:
- `data/rsa/loop_smoke2/`: the two-round run.
- `data/rsa/loop_smoke2_seeds/`: seed fits and the prune, no agents.
- `data/rsa/loop_smoke2_repair/`: a forced-repair round on pragmods.

Each holds the report, history, ledger, token usage, every step's CSVs and
each candidate's files and agent log. Not committed: `.fit_cache/`,
`responses.csv`, agent `scratch/`, `.xdg_data/`, `.home/`, and the external
cache.

## Verdict

**Nearly ready for Sherlock, but fix the memory and fit cost first (see
Recommendation 1).** Everything the handoff asked to exercise worked:

- the download;
- the combined data;
- Gemini 3.8 Flash;
- self-checks that finish and that agents read;
- a rejection followed by a successful repair;
- two rounds;
- the new pruning unit;
- snapshots off.

On the combined data, all six agent candidates of the two rounds were
admitted on the first attempt. The best model improved by 119 nats over the
best seed (`rsa_l2`, −21198.3 → `rsa_l2_valence_shared_prior`, −21079.2).

Four things went wrong, and the production plan should change because of them:

1. **The loop was OOM-killed in round 2** (14.3 GB cgroup):
   - The loop process held 3.9 GB.
   - Each agent's own fits held 1.2–1.6 GB per Python process, six at once.
   - The cause is that everything runs per trial: 50,587 trials, but only
     **268 distinct displays** (660 display × choice pairs). See
     Recommendation 1.
   - I resumed the run from disk with a scratch driver (below). No agent
     work was lost or re-run.
2. **On the combined data the self-check lies.** Two of three round-1
   self-checks reported `FAIL: the fit did not finish within 10 minutes`, for
   candidates the loop then admitted. Their agents spent their remaining
   10–20 minutes timing fits and refitting seeds, and the 40-minute limit
   killed them. In round 2 no agent ran the documented check at all.
3. **The incumbent's hypothesis misdescribes its code.** `rsa_l2_shared_prior_2`
   (best after round 1) claims depth 2 but calls `L1`. It also has a
   valence-modulated feature weight that `hypothesis.md` never mentions. The
   ledger and the round-2 briefs carried that false description.
4. **The documented self-check could not run in the sandbox** (`uv run`).
   This one is fixed, with a test.

Launching unsandboxed agents (`--no-sandbox`) was refused by this session's
permission classifier, so **all agents ran sandboxed in bubblewrap**. I
installed it with `apt-get install bubblewrap rsync`. That is the production
configuration, so this run tested more than planned. It also exposed bug 4.

## Download and data

- **Fetch and rebuild:** `src.rsa.ingest.run` fetched the four sources in
  **27 s**. Each sha256 matched.
  - The committed CSVs and provenance files rebuilt byte-identically
    (`git status` clean).
  - The licence-less `mayn_demberg_2023`/`2022` CSVs were written to
    `data/rsa/external/`, 19 MB in total.
- **Tests:** `RSA_INGEST_FETCH=1 pytest tests/test_rsa_ingest_*.py` passed
  42 tests in 35 s, with none skipped.
- **Combined data:** `combine` wrote `data/rsa/combined_trials.csv`: 61,084
  rows, 53,596 included, 27.7 MB. Of those, 50,587 are forced-choice trials
  the loop fits.

| source | rows | included |
|---|---|---|
| pragmods | 10,168 | 8,569 |
| mayn_demberg_2026 | 20,196 | 16,764 |
| mayn_demberg_2023 | 15,642 | 15,048 |
| mayn_demberg_2022 | 7,590 | 7,590 |
| sikos_2021 | 7,488 | 5,625 |

## Fit times (these size the Sherlock job)

**Seed fits on the combined data** (`loop_smoke2_seeds`, sequential, 4 CPUs).
A probe agent shared the CPUs from 01:13 to 01:16.

| model | params | fit + pool predictions |
|---|---|---|
| literal_listener | 1 | 2.7 min |
| rsa_l1 | 2 | 4.8 min |
| rsa_l2 | 2 | 6.9 min |
| rsa_l1_salience | 4 | 6.4 min |
| rsa_l1_shared_prior | 4 | 8.1 min |
| **all five** | | **28.9 min** (under the 45-min fallback, so no fallback) |

On pragmods alone (6,703 trials) the same five took 9.5 min, so 7.5× the
trials costs about 3× the time.

**Admission fits on the combined data**, for one candidate at a time, as
fit + novelty pool:

| candidate | depth | fit |
|---|---|---|
| valence_salience_listener | 1 | 7.6 min |
| rsa_l2_shared_prior_2 | 1 (named 2) | ~11 min |
| rsa_l2_shared_prior | 2 | 13.6 min |
| rsa_l2_valence_shared_prior | 2 | 17.0 min |
| valence_salience_l2 | 2 | ~22 min |

Depth-2 models take 14–22 minutes, against the 30-minute admission limit.
A depth-3 model, or a slower CPU, would be refused as "too slow".

**Scoring dominates the wall time.** `score()` recomputes PSIS-LOO for every
live model and predicts 200 draws × 50,587 trials per model. `standing()`
also runs at every round start. `RSAFit.loo()` is not cached, and `standing()`
computes it twice per model.

| scoring pass | live models | time |
|---|---|---|
| step 0 | 5 | 12.4 min |
| round-2 start (standing only) | 8 | 7.7 min |
| step 1 | 8 | 20 min |
| step 2 | 11 | 30 min |
| end (standing, prune, standing, score) | 11 → 5 | 33 min |

Even with every seed fit cached, setup and step 0 took 18 minutes.

**Phase timeline of the main run** (seed fits from the cache):

| phase | time |
|---|---|
| setup + step 0 | 02:06–02:24 |
| round 1 agents | 02:24:44–03:04:45 (40 min; two hit the limit) |
| round 1 admissions | 03:04–03:39 (34 min) |
| step 1 | 03:39–03:59 |
| round 2 start | 03:59–04:07 |
| round 2 agents | 04:07–04:20, then the OOM kill |
| resume (rebuild, standing) | 04:52–05:05 |
| round 2 admissions | 05:05–05:57 (52 min) |
| step 2 | 05:57–06:28 |
| end | 06:28–07:01 |

Loop work took about 4 h 25 min, of which about 2 h was scoring.

## Memory (why the run was killed)

At the kill (04:20:24, `dmesg`, memory cgroup limit 14.3 GB):

| process | RSS |
|---|---|
| the loop (8 live fits held in memory) | 3.9 GB (5.8 GB seen earlier during scoring; 6.5 GB with 11 models in the resume) |
| 3 × opencode | 0.5 GB each |
| 6 × agent Python processes (self-check parent + child, ad-hoc fits) | 1.2–1.6 GB each, 8.2 GB in total |

Each fit's pointwise log-likelihood is 2 × 500 × 50,587 float32 values,
about 200 MB per model. The self-check loads all trials in its parent (for
the contract) and again in its time-limited child (for the fit).

## Per slot

Round 1 standing is from step 1, round 2 from step 2. "Self-check" means the
documented `check_candidate` command. Tokens are input / cache read / output
/ reasoning. Round 1's spend is from `token_usage.jsonl`; round 2's is read
from the agents' `agent.jsonl`, because the agents finished at the moment the
loop was killed, before their records were written. Gemini 3.8 Flash latency
is the time from spawn to the first tool call. For comparison, the Pro
preview took up to 13 min 46 s in smoke test 1.

| slot | role | outcome | model | ELPD-LOO | self-check | agent wall | tokens | cost | first tool |
|---|---|---|---|---|---|---|---|---|---|
| r1 c1 | explore ("a genuinely different family") | admitted | `valence_salience_listener` | −21154.6 | 1 call, **PASS** in 611 s | 17.9 min | 379k / 3.27M / 4.5k / 31.1k | $0.66 | 2.1 s |
| r1 c2 | refine incumbent `rsa_l2` | admitted | `rsa_l2_shared_prior` | −21169.6 | 1 call, **FAIL: fit > 10 min** (769 s); read it, then timed its own fits | 40.0 min (killed at the limit, files already written) | 660k / 2.13M / 5.2k / 23.1k | $0.76 | 2.6 s |
| r1 c3 | refine chosen (`rsa_l1_shared_prior`) | admitted | `rsa_l2_shared_prior_2` (renamed: clash with c2) | −21117.1, **best after round 1** | 1 call, **FAIL: fit > 10 min** (779 s); then refitted two seeds itself (7 min each) | 40.0 min (killed at the limit, files written) | 585k / 3.18M / 5.6k / 21.1k | $0.78 | 4.2 s |
| r2 c1 | explore ("disagree most sharply with the best model") | admitted | `contrast_salience_listener` | −21099.2 | not run; its own code gate + contract script | 13.6 min | 720k / 3.89M / 7.0k / 40.5k | $1.01 | 1.2 s |
| r2 c2 | refine incumbent `rsa_l2_shared_prior_2` | admitted | `rsa_l2_valence_shared_prior` | **−21079.2, final best** | not run; its own gate + contract script | 7.8 min | 312k / 3.31M / 4.8k / 18.6k | $0.57 | 2.5 s |
| r2 c3 | refine chosen (`valence_salience_listener`) | admitted | `valence_salience_l2` | −21108.1 | not run; its own contract and **novelty RMSE vs every live model** | 10.4 min | 443k / 5.71M / 8.4k / 38.4k | $0.94 | 2.4 s |

Every slot wrote a candidate on its first spawn, so there were no retries and
no stalls. Spend for the two rounds: **$4.72** (main run, recorded $2.20 plus
$2.52 from round-2 logs).

**Repair.** In both main rounds every candidate passed, so no repair ran. To
exercise the path I ran one forced round (`loop_smoke2_repair`):

- **Setup:** pragmods only, `--candidate-count 2`, and
  `--novelty-rmse-threshold 0.03` (production: 0.002). Agents ran sandboxed.
- **c1, explore:** `valence_evaluative_listener`, admitted at −4908.2. Its
  self-check passed in 166 s. 11.4 min, $0.72.
- **c2, refine incumbent `rsa_l1_salience`:** it rebuilt smoke test 1's
  `rsa_l2_salience` almost exactly. The novelty gate rejected it at RMSE
  0.0102 < 0.03 from `rsa_l1_salience` (smoke test 1 measured 0.010). Its
  self-check passed in 203 s. 11.7 min, $0.59.
- **c2 repair:**
  - It got the reason verbatim in its prompt (`repair_note`) and its files
    copied in, then opened `novelty.py`.
  - It changed the mechanism rather than re-parameterising: depth 2 out, a
    valence-conditioned referent prior in.
  - Its self-check passed in 183 s, and it was **admitted as
    `valence_salience_listener`, −4875.2, the best model on pragmods**.
    17.2 min, $0.71.
  - **The repair path works end to end.**
- **On pragmods every self-check finished in under 3.5 min and printed PASS**,
  and each agent read it. The self-check problem is specific to the
  combined data's size.

**Memo errors.**
- The only memo error any agent hit was
  `MemoError: Python couldn't find your memo source code`. Six of nine agents
  hit it, every time from running memo code through `python -c`. Each
  recovered by writing a file.
- There were no compile or gradient errors in any submitted model.
- The other errors came from agents guessing the harness API in their own
  scripts (`TypeError: object of type 'Trials' has no len()`,
  `RSAFit has no attribute 'load'`).

## Round 2: does it build on round 1?

Yes:
- **Briefs:** round 2's briefs listed all eight live models with their
  standing, named `rsa_l2_shared_prior_2` as incumbent, and offered round 1's
  models on the refinement menu.
- **Use:** two of three round-2 slots refined round-1 models. The valence
  term that round 1's exploratory slot introduced is in both of them: one
  refined `valence_salience_listener` explicitly, the other inherited the
  term through the incumbent.
- **New ideas came from the exploratory slots:** valence framing in round 1,
  which addresses PLAN.md's "E6 valence unexplained", and display-contrast
  salience in round 2.

There is convergence, again in the two refinement slots:
- **Round 1:** both refinement slots named their model `rsa_l2_shared_prior`.
- **Round 2:** both refinement slots made valence + depth 2. c2 keeps a
  prior shared by every level, including the literal listener, over feature
  count (weighted by valence) and familiarization. c3's prior is valence ×
  feature count only and is absent from the literal listener.
- The novelty gate let all of these through, because the predictions differ
  by more than 0.002.

**The hypotheses are not reliable descriptions of the code.**
- **Round 1 c3:** its hypothesis says it "extends rsa_l1_shared_prior from
  depth 1 to depth 2". Its code still calls `L1` and adds
  `w_features + w_valence · valence`, which the hypothesis does not mention.
- **Round 2 c2:** it read the code correctly and actually made it depth 2.
  Its hypothesis ("depth 1 → 2") is right about the code, but it contradicts
  the incumbent's own description.
- **Consequence:** the ledger, `existing_hypotheses.md` and the menu describe
  the incumbent's mechanism wrongly. Nothing in the loop checks a hypothesis
  against its code.

## The end-of-run prune

- **Rule:** prune at 2 × clustered SE, the unit being display within
  condition. There are 534 clusters on the combined data.
- **Result:** six models were pruned, five kept, and the cap of 8 never
  fired.
- **Exported:** `rsa_l2_valence_shared_prior`.

| model | elpd_diff | clustered dse | ratio | outcome |
|---|---|---|---|---|
| rsa_l2_shared_prior_2 | 37.9 | 7.7 | 4.9 | pruned |
| valence_salience_listener | 75.4 | 28.2 | 2.7 | pruned |
| rsa_l1_shared_prior | 128.2 | 54.0 | 2.4 | pruned |
| rsa_l1_salience | 132.2 | 53.9 | 2.5 | pruned |
| rsa_l1 | 166.9 | 60.2 | 2.8 | pruned |
| literal_listener | 1597.6 | 253.2 | 6.3 | pruned |
| rsa_l2 | 119.1 | 59.6 | **2.00** | kept, by 0.1 nat |
| rsa_l2_shared_prior | 90.4 | 53.8 | 1.68 | kept |
| valence_salience_l2 | 28.9 | 25.1 | 1.15 | kept |
| contrast_salience_listener | 20.0 | 71.5 | 0.28 | kept |

- **The unit works.** Unlike smoke test 1, the prune fires, and the
  literal-listener and depth-1 seeds go.
- **Clustered vs trial-level SE:** the clustered SE is 3–4× the trial-level
  `dse` (e.g. 59.6 vs 16.0 for `rsa_l2`).
- **The most distinct mechanism is the least resolved.**
  `contrast_salience_listener` is 20 nats behind with a clustered SE of 71.5.
  The next design should discriminate it.
- **Seed-only run:** the same unit pruned `rsa_l1` (47.8 / 10.1, ratio 4.7)
  and `literal_listener` (1478 / 247, 6.0), and kept `rsa_l1_shared_prior`
  and `rsa_l1_salience`.

## Disk

- **Snapshots are off.** There are no `.xdg_data/opencode/snapshot`
  directories anywhere. A round-1 agent's `.xdg_data` is 6.3 MB and its
  `scratch/` 5.4 MB, against ~1 GB per agent in smoke test 1.
- **Run directory:** `data/rsa/loop_smoke2` is **994 MB**:
  - 707 MB: the three round-2 agents' private homes (`.home/.npm`, opencode's
    provider packages, ~235 MB per agent). The harness deletes a home when its
    agent exits, but the OOM kill skipped that. Round 1's homes were deleted
    as designed (round 1 total 37 MB).
  - 222 MB: the fit cache, ~21 MB per model, almost all per-trial
    log-likelihood.
  - 27 MB: `responses.csv`. It is the full 29-column provenance schema, of
    which the model uses about 8 columns.
- **Other runs:** `loop_smoke2_seeds` is 123 MB.

## Bugs

**Fixed (with a test): the self-check could not run in the sandbox.**
- **Cause:** the brief and primer documented
  `uv run python -m src.rsa.loop.check_candidate …`. In the bubblewrap
  sandbox there is no `uv`, because `~/.local/bin` is hidden by the private
  home, and `uv run` would try to sync the read-only venv. Verified inside
  the sandbox: `command -v uv` fails, while the venv's Python imports memo
  and numpyro.
- **Effect:** under the sandbox (that is, on Sherlock) every agent's
  self-check would have failed with "command not found".
- **Fix:** `brief.CHECK_COMMAND` now uses `sys.executable`, as the PyMC
  loop's `check_candidate_command` does. The primer tells agents to use the
  exact command in their `CONTEXT.md` (`src/rsa/loop/brief.py`,
  `prompts/rsa_theory.md`, `check_candidate.py` docstring).
- **Test:**
  `tests/test_rsa_loop_brief.py::test_the_self_check_runs_with_the_harness_interpreter_not_uv`.
- **Verified:** a sandboxed probe agent ran the new command and got PASS, and
  so did every sandboxed agent in this run.

**Found, not fixed** (not blocking, or larger than a small fix; see
Recommendations):
- **No resume in the RSA loop.** The OOM kill lost the process.
  - I resumed with a scratch driver, committed as
    `data/rsa/loop_smoke2/resume_round2.py`. It rebuilds `RSALoop`'s state
    from disk (the live set and manifest, fits from the content-addressed
    cache, the ledger, `history.json`, a novelty-pool digest check). It then
    runs `run_round(1)` with a spawner that reuses the three finished round-2
    agents' candidates (all written before the kill) and spawns real agents
    for any retry or repair, then `end()`.
  - The round-2 briefs it rewrote differed only in the interpreter path
    (`python` vs `python3`). I restored the originals the agents saw.
  - A first attempt lacked an `if __name__ == "__main__"` guard. The
    admission fit's spawned child re-ran the driver. I killed it before it
    wrote anything (ledger and history verified unchanged) and relaunched.
- **The ledger context of a refine-chosen slot names the incumbent, not the
  chosen model.**
  - It reads `round 1 candidate 3 refine chosen rsa_l2` for an agent that
    refined `rsa_l1_shared_prior`. The cause is that `run_round` appends the
    incumbent for every non-explore role.
  - The PyMC loop writes `… refine chosen` with no name.
- **The refinement menu is in manifest order, not ranked by standing.**
  `literal_listener`, 1,560 nats behind, comes first.
- **Agent usage is lost when the loop dies.** Records are written when
  `run_coding_agent` returns, so the round-2 agents' spend appears only in
  their logs.
- **`RSAFit.loo()` is uncached,** so `standing()` computes PSIS twice per
  model, and it is called at every round start, every score and in the end
  step.

## Recommendations for the Sherlock production run

**1. Before Sherlock: fit aggregated counts, not trials** (the largest
change; exact for every model the contract allows).
- **Why it is exact:** a model sees only the display (`Context` has no
  participant or trial fields). The 50,587 trials are 268 displays and 660
  (display, choice) pairs; per condition cluster, a few thousand cells.
- **Likelihood:** fit Σ count × log p over unique (cluster, display, choice)
  rows.
- **LOO:** compute one `loo_i` per unique row and weight it by the count.
  Identical trials have identical leave-one-out terms, so ELPD-LOO, its SE
  and the clustered SE are unchanged.
- **Expected effect** (an estimate, not measured): fits, self-checks,
  scoring and memory all fall by one to two orders of magnitude. That removes:
  - the OOM;
  - the self-check's false FAILs;
  - the 14–22 min depth-2 fits near the 30-min limit;
  - the ~2 h of scoring per two rounds.
- **Smaller wins either way:** cache `RSAFit.loo()` per fit; compute
  `cells.csv` predictions per unique display; have the self-check load the
  data once.

**2. Job size**

*If (1) is done*, a 6 × 5 run should fit easily in:
- **CPUs:** 16 (one per agent plus fits);
- **memory:** 32 GB;
- **wall time:** 12 h.

Confirm this with one round on Sherlock before the full run.

*If the run goes as the code is now* (50k trials, 6 slots, 5 rounds), size
it from this run:

- **Memory: request 64 GB.**
  - The loop is 4–7 GB and grows ~0.5 GB per live model; with 8 live plus 6
    candidates per round, ~8 GB.
  - Each agent needs ~4 GB: opencode 0.5 GB, the self-check's two processes
    2.6–3.2 GB, and the agents' own scripts, 1.3 GB each.
  - Six agents ≈ 24 GB, plus the admission-fit child ~2 GB.
  - That makes ~35 GB at peak; 64 GB gives a margin. 14 GB failed here with
    three agents.
- **CPUs: 16.**
  - The self-check (600 NUTS iterations × 2 chains) on the full data needs
    about 2 dedicated cores to finish inside its 10 minutes.
  - Here three agents shared 4 CPUs and two checks timed out at 10 min.
  - 6 agents × 2, plus 4 for the loop's admission fits and scoring.
- **Wall time: 48 h on `--qos=long`** (which refuses limits under 48 h).
  - Per round: agents ≤ 40 min; sequential admissions 6 × 8–22 min ≈ 1.5 h
    (repairs add up to 40 min plus their fits); scoring at round start and
    end ≈ 2 × 3 min × 8–14 models ≈ 1–1.4 h.
  - That is ~3.5–4 h per round, and 5 rounds ≈ 18–20 h.
  - Add seed fits (30 min) and the end step (~1 h): ~22 h. Doubled for
    Sherlock's CPUs and retries, that is 48 h.
- **Raise the admission time limit to 45–60 min** (`--fit-time-limit-sec`).
  Depth-2 fits took up to 22 min on this container; a depth-3 model would be
  refused.

**3. Agent settings**
- **Model and timeout:** keep `google/gemini-3.8-flash` and
  `--agent-timeout-sec 2400`.
  - Flash answered in 1–4 s, never stalled, and wrote a candidate on every
    one of nine spawns.
  - It cost $0.57–1.01 per agent, ~$0.8 on average. Budget about $5 per
    round of 6, ~$25–30 per 5-round run with repairs.
- **Sandboxing:** keep agents sandboxed. This run shows sandboxed opencode
  agents work end to end with the corrected self-check command.
- **Self-check data:** if aggregation is not done first, run the self-check
  on a fixed subsample, e.g. pragmods or one display per condition, and say
  so in the primer. On the full data it reports false FAILs and tempts
  agents into their own 7-minute fits.
- **Primer additions**, all of which agents hit:
  - memo code must live in a file (`python -c` raises "couldn't find your
    memo source code");
  - do not refit the existing models;
  - the self-check needs one run.

**4. Fix before production** (loop correctness):
- **Hypothesis–code consistency.** The refinement brief should ask for the
  hypothesis to list every change to the parent's code, its depth and its
  parameters. A cheap mechanical check would help too: compare the
  candidate's `PARAMS` and the `L<k>` its `choice_probs` calls against the
  hypothesis text, or show the agent a diff from its parent.
  `rsa_l2_shared_prior_2` reached the top of the ledger with a false
  description.
- **The refine-chosen context:** stop appending the incumbent's name.
- **The refinement menu:** rank it by standing.
- **Resume support in `src/rsa/loop`:** a `--resume` that rebuilds state from
  disk as `resume_round2.py` does. A multi-hour Sherlock job needs it.
- **Usage records:** record agent usage even when the parent dies, by
  writing the record from the agent's own log after a crash.
- **npm cache:** share one read-only npm cache across agents (opencode
  downloads ~235 MB into each private home), or at least delete homes left
  by a crashed run on the next start.

**5. Convergence of the refinement slots.** In both rounds the two
refinement slots converged on one idea. With 6 slots (two refining the
incumbent) expect pairs of near-twins. The novelty threshold (0.002) admits
them, and the prune later removes one of each pair. That is acceptable, but
it spends a slot. If it matters, steer the second incumbent-refinement slot
with a lens ("change the speaker", "change the prior").
