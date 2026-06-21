# Auto-psych — Problems & Resolutions

Findings from a code review of `src/` (~18k LOC), now annotated with how each was
resolved. **Status** is one of:

- **FIXED** — code changed; behavior corrected.
- **DOCUMENTED** — a real concern, but the "fix" is a scientific/representation
  decision left to the researcher; the code now states the caveat clearly.
- **DEFERRED** — intentionally not changed (the current behavior is defensible or
  the change would alter scientific output); rationale given.

All changed files pass an AST syntax check, and the pure-logic helpers (BH-FDR,
reward floor, choice coercion) were verified in isolation. The full `pytest`
suite must be run in the project environment (`uv run --group dev pytest tests
src/pipelines/inner_loop/tests -q`) — it could not run on the login node (Python
3.6, no project deps). Changes were checked against the existing tests'
expectations by inspection (`test_critique_ppc.py`, `test_deployment_firebase.py`,
`test_deployment_firestore.py`).

---

## HIGH severity

### H1. Critique PPC null drawn from a single MCMC chain — **FIXED**
`src/models/pymc_inference.py` (`sample_synthetic_responses`). Replaced the
chain-major head slice `flat[:n_datasets]` with an even stride
(`np.linspace(0, N-1, n_datasets)`) across the full chain×draw pool, so the PPC
null distribution spans all chains/draws and is reproducible.

### H2. `update_registry_from_interpretation` discarded the computed posterior — **FIXED**
`src/pipelines/outer_loop/orchestrator.py` now writes the real `posteriors`
(renormalized, finite-filtered) into the registry instead of a hard-coded
`{"inner_loop_model": 1.0}`. `src/pipelines/outer_loop/run.py`
(`_posterior_design_inputs`) now aligns the previous experiment's weights to the
current model set (intersection + renormalize), falling back to a uniform prior
(with a log line) when names don't overlap — so the exhaustive posterior-design
path for experiments ≥2 no longer crashes or degenerates to a single model.

### H3. Viewer served arbitrary files from any run dir — **FIXED**
`src/viewer/server.py`: the `/files/<rel>` route now enforces an extension
allow-list (`_SERVABLE_SUFFIXES`: images + html/css/js) checked *before* touching
the filesystem, returning 403 for anything else (`.py`, `.jsonl`, `.json`,
`.yaml`, `.nc`, …). Added a JSON 403 handler.

### H4. Unauthenticated `0.0.0.0` exposure — **FIXED**
`src/viewer/server.py` and `src/monitor/server.py` now print a loud warning when
bound to a non-loopback host, naming exactly what is exposed (run artifacts /
live participant data). (Authentication itself is out of scope for a local
research tool; the warning makes the risk explicit.)

### H5. Coding-agent timeout leaked grandchildren / could hang — **FIXED**
`src/runtime/coding_agent.py`: `Popen(..., start_new_session=True)` plus
`os.killpg(os.getpgid(pid), SIGKILL)` on timeout, so the whole agent process tree
is killed (and the stdout read loop unblocks) instead of leaking children and
hanging `proc.wait()`.

### H6. Participant reward could compute to 0 — **FIXED**
`src/pipelines/outer_loop/deployment/prolific.py` (`compute_reward_cents`) now
raises on a non-positive `reward_per_hour`/`estimated_completion_time` or a final
reward ≤ 0, surfacing the problem at dry-run time before any study is created.

---

## MEDIUM severity

### M1. Divergent fit fingerprints (critique vs. comparison) — **FIXED**
`src/models/pymc_inference.py`: the on-disk `.nc` fingerprint now hashes the model
source + the responses-file bytes — the same inputs as the in-process
`_cache_key` — so the seeded critique always reuses the exact fit the comparison
scored. `observed` is now extracted only on a cache miss.

### M2. ELPD-softmax model posterior is overconfident (ignores ELPD SE) — **DOCUMENTED**
`src/model_comparison/posterior.py`: added an explicit caveat to
`model_posterior` explaining that the softmax of total ELPD-LOO yields a near
one-hot vector that ignores the ELPD standard error, and that `compare_table`'s
`dse` column is the SE-aware view. The math is unchanged (switching to
stacking/SE-aware weights is a research decision left to you).

### M3. Inconsistent CSV reading across the codebase — **FIXED (partial)**
Row counts now use `csv.DictReader` with properly closed handles in
`src/model_comparison/posterior.py` and `likelihood.py`. The deeper
`csv.DictReader` (fit) vs. `pandas.read_csv` (critique) unification is **DEFERRED**
(both read the same on-disk file in the active flow, so they agree in practice);
documented as a latent risk.

### M4. `model_logp_is_finite` only checked the initial-point logp — **FIXED**
`src/models/pymc_inference.py` now also evaluates `compile_dlogp()` and rejects a
non-finite gradient at the initial point (the common NUTS-abort cause a logp-only
check missed). Still not a full guarantee (documented).

### M5. Agent code `exec`'d with no timeout — **FIXED (best-effort)**
`src/critique/ppc.py`: test-statistic evaluation is wrapped in a best-effort
`SIGALRM` wall-clock limit (`_TEST_STAT_TIMEOUT_SEC = 30`, main-thread Unix only),
so a runaway statistic can't hang the in-process harness. Full sandboxing of
builtins is **DEFERRED** (accepted risk for a trusted single-user pipeline).

### M6. No multiplicity control over 8 screened statistics — **FIXED**
`src/critique/ppc.py` now reports a Benjamini-Hochberg FDR-adjusted q
(`p_value_fdr`) and flag (`significant_fdr`, `n_significant_fdr`) **alongside** the
raw p. `src/pipelines/inner_loop/pymc_orchestrator.py` renders the q in
`critiques.md` and softens the candidate prompt from "strongest evidence" to
"prefer discrepancies that survive the FDR." (The existing test already
anticipated BH with m=2.)

### M7. Tempfile/file-descriptor leaks — **FIXED**
`src/model_comparison/posterior.py`: `mkstemp` fd is closed, the pooled temp file
is unlinked after use (try/finally in `main`), and the `n_trials` reader is
context-managed. `likelihood.py` likewise context-manages its reader.

### M8. Live collection could skip the quality gate; brittle DictWriter — **FIXED**
The active writer (`src/pipelines/outer_loop/orchestrator.py`
`run_collect_programmatic`, which already gates via `check_response_variation`)
now writes the **union** of all row keys with `restval=""`, so heterogeneous
live/Firebase rows don't raise or drop data. (`collect.run_collect` is dead code —
no callers; flagged for removal.)

### M9. Non-atomic Firestore metadata write — **FIXED**
`src/pipelines/outer_loop/deployment/firestore.py` (`write_metadata`) now writes
the study/deployment/session docs in a single `WriteBatch` (all-or-nothing).

### M10. Firebase deploy serialization was opt-in — **FIXED**
`src/pipelines/outer_loop/deployment/firebase.py`: `_deploy_lock(project)` now
defaults to a per-project lockfile in the system temp dir (serializes same-machine
deploys automatically); `AUTO_PSYCH_DEPLOY_LOCK` still overrides for cross-node
(shared-filesystem) serialization.

### M11. Consent/submit verification was a substring check — **FIXED**
`src/pipelines/outer_loop/deployment/firebase.py`: `_has_working_submit` detects an
actual `fetch(... "/submit")` (or the injected marker) instead of a bare
substring; the bridge carries a `SUBMIT_BRIDGE_MARKER` (idempotent injection); and
`stage_experiment` now asserts a working submit path post-staging, turning silent
no-collection into a loud `DeploymentError`.

### M12. Monitor miscounted string `chose_left` — **FIXED**
`src/monitor/aggregate.py` (and the same bug in
`deployment/firestore.py::responses_to_csv`) now coerce via an explicit
`_chose_left` helper, so `"0"`/`"false"`/`"right"` are no longer counted as left.

### M13. EIG-vs-random arms shared one RNG stream — **FIXED**
`src/subjective_randomness/adaptive_recovery.py`: response noise is now seeded by
`[seed, repeat, arm_index]` (and `gen_index` in `compare_model_recovery`), giving
each arm/generating-model independent, reproducible Bernoulli draws.

### M14. `get_model_predictions` silently dropped models — **FIXED**
`src/models/theorist/predictions.py` now catches broadly (incl.
`TypeError`/`AttributeError`) and logs the dropped model + reason instead of
silently omitting it.

### M15. `datetime.utcnow()` participant-ID collisions — **FIXED**
`src/pipelines/outer_loop/collect.py`: new `_unique_batch_id()` uses a
timezone-aware UTC clock with microseconds + a short uuid token, so two passes in
the same second no longer produce colliding participant IDs.

---

## LOW severity

- **L1 periodicity_score discontinuity / truncated period range — DEFERRED.**
  `features.py` + `model_families/common.py`. Changing the feature's scaling alters
  model inputs (a scientific decision). The duplicated copies remain a
  maintainability risk worth consolidating.
- **L2 "chars" vs "lines" docs — FIXED** (`model_comparison/posterior.py` docstrings + CLI help).
- **L3 `p_value_is_floor` at ties — DEFERRED.** It is self-consistent with the
  two-sided p formula as written (which uses `n_ge`/`n_le` with equality).
- **L4 re-`exec` per replicate — FIXED.** `critique/ppc.py` compiles the statistic
  once (`_compile_test_statistic`) and calls it many times.
- **L5 `_thin_posterior` kept earliest draws / under-counted — FIXED.** Now thins
  by an even stride across each chain.
- **L6 pydantic/`TypeError` escaped the viewer error handler — FIXED.**
  `viewer/scan.py` wraps `TrajectoryStep(**step)` and guards `max(posteriors,…)`
  (`_argmax_posteriors`), raising a `ValueError` naming the offending artifact.
- **L7 design EIG floor (`max<=0`) — DEFERRED.** Tightening the threshold risks
  rejecting legitimate designs; left as-is.
- **L8 swallowed exceptions — FIXED.** `validation/validators.py` and
  `experiments/references.py` now log instead of silently passing/continuing.
- **L9 unseeded `random` in `_generate_from_models` — FIXED.** Now uses a seeded
  local `random.Random(seed)`, like the PyMC path.
- **L10 misc robustness — FIXED:** `build_command` raises (no `None` into `Popen`);
  `.secrets` values are quote-stripped (`llm.py`, `runtime/prolific.py`);
  `parse_rating` rounds instead of truncating; the discriminating-probe control set
  is now disjoint from the adversarial set. (`parse_problem_definition` reverse
  ranges left as-is.)

---

## Documentation & repo hygiene

- **D1 stale README — FIXED.** The intro no longer references a non-existent
  `legacy/`; the Project Layout now matches the real tree (`pymc_orchestrator.py`,
  `runtime/coding_agent.py`, `critique/`, `model_comparison/`,
  `subjective_randomness/`, `deployment/`).
- **D2 default-backend mismatch — FIXED.** Per your decision, `opencode` stays the
  default; the README and both `run.py` `--coding-agent` help strings now say
  `opencode` (default model `google/gemini-3.1-pro-preview`), with `claude` as the
  opt-in.
- **D3 repo hygiene — PARTIAL.** `gt.txt` (~21 MB) added to `.gitignore` with an
  untrack note (run `git rm --cached gt.txt` to drop it from history going
  forward). The duplicated `projects/` trees, loose root scripts
  (`candidate.py`/`extract_pdf.py`/`generate_candidates.py`), and stale
  `CLEANUP_PLAN.md` were **not** deleted (pre-existing tracked files you own —
  surfaced here for you to prune).

---

## Verified correct (not bugs)

No shell-injection (list-form argv, `shell=False`); viewer path traversal is
blocked (`_run_dir` resolve + containment, `safe_join`); the `model_posterior`
softmax is numerically stable; the `+1` empirical-p correction and log-domain
recovery updates are sound; sequential Bayesian model evidence in
`adaptive_recovery` is computed correctly. `.secrets` itself is git-ignored
(only `.secrets.example` is tracked).
