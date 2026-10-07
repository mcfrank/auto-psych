# RSA loop smoke test with real Gemini agents: results

Run 2026-10-06 23:14 to 2026-10-07 00:11 UTC (56 min), in a disposable
claude.ai cloud container (4 CPUs), on branch `auto-rsa-smoke` (from
`auto-rsa` at `c1dd78f`), following `HANDOFF_smoke_test.md`:

```bash
uv run python -m src.rsa.loop.run --results data/rsa/loop_smoke \
    --max-iterations 1 --candidate-count 3 --no-sandbox \
    --num-warmup 500 --num-samples 500 --num-chains 2
```

Agents: opencode 1.18.35, `google/gemini-3.1-pro-preview` (the default).
Data: all 6,703 included forced-choice pragmods trials. Run outputs that are
small are committed under `data/rsa/loop_smoke/` (report, history, ledger,
each candidate's files and agent logs; not `.fit_cache/`, `responses.csv`
or opencode's `.xdg_data/` snapshots).

## Verdict

**Success on the handoff's criterion. The self-check did not work in this run,
and the repair path was never exercised.**

- Two of three slots wrote a memo model that passed every gate (code, contract,
  fit, convergence, finite ELPD, novelty) on the **first** attempt. One of
  them, `rsa_l2_salience`, is the new best model: −4955.5, which is
  17.7 ± 2.7 nats ahead of the previous best seed `rsa_l1_salience`.
- Neither agent ever saw a self-check result. Every self-check call failed: on
  a CLI bug (now fixed) or on opencode's 120 s shell timeout (fixed in the
  brief and primer). Their models passed because they stayed close to the
  seed files' memo, not because the self-check caught anything.
- One slot (refine incumbent) wrote no `candidate.py`, twice. No candidate was
  rejected, so **no repair ran**. Whether the rejection reasons let a repair
  succeed is still untested.
- The agents hit **no memo errors at all**. Each one read seed files first and
  changed them in small steps (a new prior, an extra level). The primer's
  gotchas were not tested in anger.

## Per slot

| Slot | Role | Outcome | Model | ELPD-LOO | Novelty: nearest (RMSE) | Self-check | Agent wall time | Tokens (in / cached / out / reasoning) | Cost |
|---|---|---|---|---|---|---|---|---|---|
| 1 | explore (lens: "a genuinely different family") | **admitted** | `literal_speaker_inverter` | −5046.1 (90.6 ± 15.3 behind) | `rsa_l1_shared_prior` (0.045) | 3 calls: usage error, then killed at 120 s twice | 12 min | 77k / 63k / 0.8k / 15.1k | $0.36 |
| 2 | refine incumbent (`rsa_l1_salience`) | **no file** | (drafts in `/tmp`, never in its dir) | — | — | 1 call, killed at 120 s | 20 min (killed at the limit) | 57k / 152k / 0.9k / 2.0k | $0.18 |
| 2 retry | same | **no file** | — | — | — | none | 20 min (killed at the limit) | not reported (killed) | ? |
| 3 | refine chosen | **admitted** | `rsa_l2_salience` | **−4955.5 (best)** | `rsa_l1_salience` (0.010) | 3 calls: usage error, then killed at 120 s twice | 19 min | 39k / 256k / 1.1k / 3.0k | $0.18 |

Reported spend: **$0.71** for the round. The killed retry reported no usage,
so the true figure is a little higher. Token use per slot is small: ~33k of
each prompt is the brief, primer and handbook, which caching mostly absorbs.

Phase timing: seed fits 6.5 min (5 models); agents 20 min (bound by the
agent timeout of slot 2); the retry 20 min (killed again); admission fits of
the two candidates 4 min; scoring and report 4 min.

### Slot 1: `literal_speaker_inverter` (explore)

> Listeners invert a zero-order "literal" speaker (S0) who simply picks one
> of the object's features uniformly at random, without simulating a
> listener. The pragmatic preference for simpler objects emerges entirely
> from probability dilution.

This is a genuinely different mechanism: L1 over a literal speaker, with no
informativity term, and a prior on familiarization, grayscale and feature
count. It is 90.6 nats behind the best model, yet clearly better than the
literal listener. It is a useful contrast: the data favour a speaker who
weighs informativity. The agent spent its first step on 15k reasoning tokens
and wrote all three files in one heredoc, and its memo compiled first time.

### Slot 2: refine incumbent, no candidate (twice)

The first agent read the seeds and drafted a depth-2 salience model in
`/tmp/cand_l2/` and in `test_l2_salience.py` **in the repository root**. It ran
the self-check on `/tmp/cand_l2`, which was killed at 120 s. It then pivoted to
an L1 + grayscale-prior variant (`/tmp/cand_l1_color/`, `test_l1_color.py` in
the repo root), and was killed at the 1,200 s agent limit while that check
ran. It never wrote to its candidate directory.

The retry agent's first event came 13 min 46 s after spawn (one `cat` of a
seed file); it then went silent until the 20 min limit killed it. This looks
like Gemini preview latency or a stalled request, not anything in the brief.

Under bubblewrap (production) the stray files in the repo root would have
failed (read-only tree), and so would the `/tmp` drafts once the sandbox was
gone. Either way the slot would be empty.

### Slot 3: `rsa_l2_salience` (refine chosen)

> I refine the `rsa_l2` model by giving its top-level pragmatic listener the
> learned salience prior over objects (based on feature count and
> familiarization) …

This is a single stated change, the refinement brief's intended use: the
salience prior of `rsa_l1_salience` grafted onto the top level of `rsa_l2`.
It is the round's discovery. The best model now has a depth-2 listener *and*
a salience prior, and leads by 17.7 nats (6.5 SE). Its novelty margin is small
but clear: RMSE 0.010 from `rsa_l1_salience`, against a threshold of 0.002.
Slot 2's abandoned draft was nearly the same model (the prior at every level),
so the two refinement slots converged on one idea.

## Bugs found and fixed (each with a test)

1. **The documented self-check command did not parse.**
   - **Cause:** The brief and primer give
     `check_candidate <dir> --responses <csv>`, but the tyro CLI declared
     `candidate_dir` as a required `--candidate-dir` flag.
   - **Effect:** All three agents ran the documented form first and got a
     usage error.
   - **Fix:** `candidate_dir` is now positional
     (`tyro.conf.Positional`, `src/rsa/loop/check_candidate.py`).
   - **Test:** `tests/test_rsa_loop_brief.py::test_the_documented_self_check_command_parses`
     parses the command exactly as `CHECK_COMMAND` writes it.
2. **The self-check never finished inside an agent.**
   - **Cause:** opencode's shell tool kills a command after 120 s by default.
     The self-check takes 2 min 11 s on an idle 4-CPU container, and longer
     while three agents run checks at once.
   - **Effect:** All five correctly formed self-check calls were killed
     before printing a verdict.
   - **Verified:** opencode's shell tool accepts a `timeout` parameter, and a
     150 s command with `timeout: 300000` completed.
   - **Fix:** `CONTEXT.md` (`brief.CHECK_SHELL_TIMEOUT_MS` = 900,000 ms) and the
     primer's self-check section now say to pass that timeout, to run the
     check once per change, and to write and test files **only in the
     candidate directory** (slot 2's failure).
   - **Test:** asserted in `test_each_role_gets_its_documents_and_the_handbook`.
3. **No `token_usage.jsonl`.**
   - **Cause:** The handoff says the launcher writes it, but
     `src/rsa/loop/run.py` never started the usage log.
   - **Effect:** This run's spend above is read from the agents' own logs.
   - **Fix:** The CLI now starts the usage log at
     `<results>/token_usage.jsonl` and writes `token_usage_summary.json` in a
     `finally`, as the PyMC loop does.
   - **Test:** `tests/test_rsa_loop_run.py`.

## Other findings and recommendations

**Recommendations, in priority order:**

1. **Raise `--agent-timeout-sec` from 1,200 to ~2,400 for production.** With
   a working self-check, a careful agent now runs 1 to 3 checks of 2 to 10 min
   each. 20 min would end many agents mid-check. Slot 3 already used 19 min,
   mostly waiting on killed checks.
2. **Turn off opencode's snapshots for loop agents.** opencode keeps a git
   snapshot of its working directory in each agent's private `XDG_DATA_HOME`
   (`<candidate>/.xdg_data/opencode/snapshot`). With the agents running from
   the repo root, that was ~1 GB per agent: **3.9 GB for this four-agent
   round**. At 5 rounds × 6 slots plus retries and repairs on Sherlock, that
   is 30 to 60 GB per run. The scrubbed agent tree will be smaller, but not
   free. opencode's config has `"snapshot": false`. Set it in the launcher's
   per-agent opencode config (`src/runtime/coding_agent.py`, shared with the
   PyMC loop, so not changed here), or delete `.xdg_data/opencode/snapshot`
   when the agent ends.
3. **Pruning at 2 × clustered dse prunes nothing on pragmods.** The end-of-run
   prune kept all 7 models, even `literal_listener` at 572 nats behind. The
   stimulus-clustered SE over the 59 display clusters is large: 331.6 for
   `literal_listener`, a ratio of 1.73. Every model's ratio to the best was
   between 1.2 and 1.7. A few large, heterogeneous clusters (displays shared
   across experiments) dominate the variance. Pruning works as written, but on
   this data it never fires, so the live set will hit the cap of 8 by round 2
   and the cap will do all the selection. Consider clustering by display
   *within* experiment, a lower multiplier, or a per-cluster sign test.
   Decide before the production run; it is a design choice, not a bug.
4. **Re-run the smoke test** with the fixes, before Sherlock. It should take
   ~1 h; consider `--candidate-count 3 --max-iterations 2` so that round 2
   sees a live set with agent-written models. Look for: (a) agents now
   reading a PASS/FAIL line; (b) at least one rejection, so the repair path
   runs. If none comes naturally, a deliberately strict novelty threshold for
   one round would force rejections.
5. **Gemini latency.** The retry agent took 13 min 46 s to make its first tool
   call. If that recurs, try a Flash model for retries (`--agent-model`), or
   accept the cost of an empty slot.
6. **The two refinement slots converged** on "L2 + salience". With 6 slots
   (two refining the incumbent) that will happen more. The PyMC loop has the
   same structure, so watch it in production rather than change it now.

**No new memo gotchas.** No agent hit a memo compile or runtime error, so the
primer's list in `rsa_theory.md` and PLAN.md stands unchanged. The agents
worked around the language by copying the seeds' memo verbatim and changing
`choice_probs` and the prior. That is safe, but it may keep exploration close
to the seeds. An exploratory slot that needs new memo structure (a lexical
uncertainty model, a QUD speaker) is still untested.

**Hypothesis style.** Slot 3's `hypothesis.md` reads as a change log ("I refine
`rsa_l2` by …") rather than a claim a psychologist could test. The refinement
menu asks the agent to name the model it refines, so this follows the brief.
A sentence in the refinement brief could ask for the claim first and the
lineage second.
