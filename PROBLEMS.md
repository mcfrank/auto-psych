# Auto-psych — Problems & Critiques

Findings from a code review of `src/` (~18k LOC). Each item gives a
`file:line` anchor, what's wrong, why it matters, and a severity
(**HIGH** / **MEDIUM** / **LOW**). Items marked *(verified)* were traced directly
during this review; the rest were surfaced by focused sub-reviews with concrete
line references and are worth confirming before acting.

The highest-priority items are: the **single-chain posterior-predictive null**
(corrupts every critique p-value), the **registry update that discards the real
model posterior** (breaks posterior-weighted design for experiments ≥2), the
**unauthenticated file/participant-data web servers**, the **coding-agent
subprocess child leak**, and the **zero/under-payment path for real
participants**.

---

## HIGH severity

### H1. Critique PPC null distribution is drawn from a single MCMC chain *(verified)*
`src/models/pymc_inference.py:546-548`

```python
arr = pp.posterior_predictive[response_rv_name].values  # (chain, draw, n_stim)
flat = arr.reshape(-1, arr.shape[-1])                    # (chain*draw, n_stim)
return flat[:n_datasets]
```

`reshape` lays samples out chain-major, so `flat[:n_datasets]` takes the *first*
`n_datasets` rows — i.e. only chain 0's first draws. The default fit is
`chains=4, draws=2000` (8000 samples) but the critique uses
`CRITIQUE_PPC_REPLICATES = 200`, so **every** posterior-predictive null dataset
comes from a single chain's first 200 draws. This biases the null distribution
that every critique p-value is computed against, wastes the other chains, and
uses autocorrelated early draws. Subsample across chains/draws (random selection
with the seed, or stride).

### H2. `update_registry_from_interpretation` throws away the computed posterior *(verified)*
`src/pipelines/outer_loop/orchestrator.py:995-1013`

The function loads `model_posterior.json`, validates that `posteriors` is a
dict, and then **ignores it**, writing a hard-coded `{"inner_loop_model": 1.0}`:

```python
if not isinstance(data.get("posteriors"), dict):
    return
write_registry(registry_path, {"inner_loop_model": 1.0}, reserved_for_new=0.0)
```

Consequences: (a) the per-experiment `model_registry.yaml` never reflects the
real model weights the inner loop computed; (b) for experiments ≥2,
`_posterior_design_inputs` (`run.py:187-214`) reads
`weights = registry.get("theories")` → `{"inner_loop_model": 1.0}`, but the
model *names* and `param_sets_by_model` come from the **current** experiment's
manifest. The weight keys therefore don't match the models being scored, so the
"posterior-weighted EIG design" the docstring promises is built on mismatched /
degenerate weights. This silently undermines the headline claim that each
experiment is designed under the previous one's posterior.

### H3. Viewer serves arbitrary files from any run directory
`src/viewer/server.py:77-83`

```python
@app.get("/api/run/<path:run_path>/files/<path:rel>")
def api_run_file(run_path, rel):
    run_dir = _run_dir(run_path)
    ...
    return send_from_directory(run_dir, rel)
```

No extension/type allow-list. Run dirs contain agent transcripts (`*.jsonl`),
raw `candidate.py` source, `config.json` (deployment URLs/metadata), etc. Path
traversal itself is blocked (`_run_dir` resolves + checks containment, and
Werkzeug's `safe_join` blocks `../`), but *everything under the run tree* is
readable. Restrict to the known asset types the frontend actually requests
(`.png/.svg/.html/.csv`).

### H4. Viewer & monitor `0.0.0.0` binding exposes unauthenticated participant data
`src/viewer/server.py:99,107`, `src/monitor/server.py:97,108`

Both servers document `--host 0.0.0.0` and have no auth, CORS, or rate limiting.
The monitor serves live human-subjects data (Prolific PIDs, per-participant
choices via `/api/session/<id>`); combined with H3 the viewer serves arbitrary
files. At minimum warn loudly in the help text; ideally gate behind a token.

### H5. Coding-agent timeout kills only the direct child; grandchildren leak / can hang *(verified)*
`src/runtime/coding_agent.py:147-186`

```python
def _kill_after():
    timed_out.set()
    proc.kill()           # SIGKILL to the direct child only
...
for raw_line in proc.stdout:   # blocks until the pipe closes
```

`claude`/`opencode`/`npx` spawn their own child processes. `proc.kill()` does not
use a process group (`start_new_session=True` + `os.killpg`), so grandchildren
survive a "timeout" and keep running on the (shared HPC) node. Worse, if a leaked
grandchild keeps the stdout pipe open, the `for ... in proc.stdout` loop and the
following `proc.wait()` can block indefinitely despite the timer firing. This is a
real resource-leak / hang risk for long agentic runs.

### H6. Real-participant reward can compute to 0 / underpay
`src/pipelines/outer_loop/deployment/prolific.py:34-37`

```python
if cfg.get("reward_per_hour") is not None:
    minutes = float(cfg.get("estimated_completion_time") or 5)
    return round(float(cfg["reward_per_hour"]) * minutes / 60.0)
return int(cfg.get("reward") or 50)
```

No positivity floor. If `estimated_completion_time` is `0`, the reward rounds to
`0` cents (Prolific rejects it, or workers are underpaid). Since this pays real
people, validate the computed reward is a sensible positive minimum.

---

## MEDIUM severity

### M1. Two separate fingerprint/cache key spaces can make the critique score a different fit than the comparison ranked
`src/critique/ppc.py` + `src/models/pymc_inference.py:554-555, 583-593`

The in-process `_FIT_CACHE` is keyed on `(name, sha256(model.py), sha256(csv bytes))`,
while the on-disk `.nc` fingerprint hashes the *bound observed arrays*
(`_sha256_dict_arrays(observed)`), not the raw CSV bytes. Two CSVs that parse to
identical arrays but differ in bytes (whitespace/column order) collide on disk
but not in process, and vice-versa — so the seeded incumbent fit can be silently
bypassed and refit with fresh MCMC, producing a *different* posterior than the
one the model comparison ranked. Unify on one fingerprint.

### M2. Bayesian "posterior over models" is effectively always one-hot and ignores ELPD standard error
`src/model_comparison/posterior.py:200-220`

`P(model|data) ∝ exp(elpd_loo) · P(model)`. ELPD-LOO is a sum over all trials, so
`exp(elpd_diff)` of a few nats across hundreds of trials makes one model ≈1.0 and
the rest ≈0.0, with no use of the ELPD difference's standard error (`dse`, which
the code computes in `compare_table` but never feeds in). The same `report.md`
then prints a "distinguishable from best (elpd_diff > 2·dse)" column that will
routinely say *no* while the posterior says ≈1.0 — internally contradictory and
overconfident.

### M3. Naive CSV reading diverges across the codebase (DictReader vs pandas vs line-count)
`src/models/pymc_inference.py:264` (`csv.DictReader`) vs `src/critique/ppc.py:246`
(`pd.read_csv`); line-count counts in `src/model_comparison/posterior.py:222`,
`likelihood.py:93`, `src/subjective_randomness/holdout_recovery.py:750`; naive
`split(",")` header parsing in `orchestrator.py:940-943`,
`src/validation/stages/collect.py:17-28`, `analysis.py:18-22`.

The fit uses `csv.DictReader`; the critique builds frames with `pandas`; trial
counts are computed by raw line count (`sum(1 for _ in open(...))`). These can
disagree on row count (trailing newline, NaN handling, embedded newlines in a
quoted field), so the PPC can condition on a different design than the model was
fit on, and reported `n_trials` can be wrong — all with no error raised. Use one
CSV reader everywhere.

### M4. `model_logp_is_finite` gate is weaker than advertised
`src/models/pymc_inference.py:319-325`

It checks logp only at the single default initial point. `pm.sample` also checks
the *gradient* and jittered initial points across chains; a model that passes
this gate can still NaN and abort the whole run. The `_drop_unfittable_models` /
`_admit_candidate` "won't crash MCMC" guarantee is therefore not airtight. Also
evaluate `compile_dlogp`, or wrap the actual sample in try/except.

### M5. `exec`/`importlib` of agent-authored code with no sandbox or timeout
`src/critique/ppc.py:140-142` (test statistics), `src/models/pymc_inference.py:71`
+ `src/models/theorist/loader.py:34` (model files), `src/validation/stages/theory.py:55-59`

Agent-written Python is `exec`'d / imported and called with full builtins and no
resource/time limit, in-process. The in-process PPC harness in particular runs
agent code *outside* the agent's subprocess `agent_timeout_sec`, so an infinite
loop or heavy I/O in a test statistic hangs the run. `get_model_callable` also
*falls back to the first callable in the module* if the expected name is absent,
so validation can silently bind to (and "pass") the wrong function. Acceptable for
a trusted single-user pipeline, but add at least a timeout.

### M6. No multiple-comparisons control while screening 8 statistics, then framed as "strongest evidence"
`src/critique/ppc.py:25-26, 372-374`; `pymc_orchestrator.py:50-54, 330-334`

Each round proposes `CRITIQUE_N_PROPOSALS = 8` statistics tested at raw `α=0.05`
with no FDR/Bonferroni (documented as intentional "exploratory screening"). But
the flagged discrepancies are then handed to candidate agents as the
**"strongest evidence"** for what to fix. With 8 tests at 0.05 the per-round
false-positive rate is ~34%, so candidate generation will frequently chase noise.
Report a BH-FDR-adjusted flag alongside the raw one and soften the framing.

### M7. Tempfile / file-descriptor leaks
`src/model_comparison/posterior.py:94` (`tempfile.mkstemp(...)[1]` — fd never
closed, temp file never unlinked); `posterior.py:222` / `likelihood.py:93`
(`csv.DictReader(path.open(...))` — handle never closed). On the cluster these
accumulate in `$TMPDIR`.

### M8. Live-collection path skips the degenerate-data quality gate; two collect drivers disagree
`src/pipelines/outer_loop/collect.py:516-543` vs `orchestrator.py:489-496`

`orchestrator.run_collect_programmatic` calls `check_response_variation` on
collected rows, but `collect.run_collect` (a separate driver) does not — so
degenerate all-one-side live data can pass through that path unflagged. Live rows
from `_collect_live`/`_collect_from_firebase` are also written via
`csv.DictWriter(fieldnames=list(rows[0].keys()))` (`collect.py:540`,
`orchestrator.py:500`); heterogeneous columns across rows raise `ValueError` or
drop data. Validate schema and route all collection through one gated path.

### M9. Firestore deployment metadata is written non-atomically
`src/pipelines/outer_loop/deployment/firestore.py:74-80`

The study/deployment/session docs are written in a plain loop with `merge=True`
and no batch/transaction. A crash mid-loop leaves a deployment referencing a
session doc that was never written, and `local.py:102-111` swallows the failure
as a warning and **publishes the Prolific study anyway** — a live study pointing
at missing Firestore state. Use a `WriteBatch`.

### M10. Firebase deploy serialization is opt-in; default concurrent deploys can clobber each other
`src/pipelines/outer_loop/deployment/firebase.py:329,357,407-419`

`_deploy_lock()` is a no-op unless `AUTO_PSYCH_DEPLOY_LOCK` is set (and `fcntl`
imports). By default, parallel runs sharing one Firebase site deploy
`functions,firestore` with `--force` and rewrite project-wide `firestore.rules`
with no serialization, so concurrent deploys race. Make the lock the default for
shared sites.

### M11. Consent / submit-bridge verification is a substring check (silent data loss)
`src/pipelines/outer_loop/deployment/firebase.py:152-200, 216-249`

`ensure_submit_bridge` is skipped when `CLIENT_CONFIG_FILENAME in index_html and
"/submit" in index_html` — a mere substring match (a comment mentioning them
suffices). The staging code asserts only the consent marker, never that a working
submit path exists, so a deployed study can silently collect nothing while
appearing fully staged.

### M12. Monitor miscounts string-valued `chose_left`, defeating its own degenerate-data detector
`src/monitor/aggregate.py:60`

```python
n_left = sum(1 for t in valid if bool(t["chose_left"]))
```

`chose_left` comes from Firestore. `bool("0")` and `bool("false")` are both
`True`, so a right-choice stored as a string is counted as left. This silently
inverts the choice balance the module exists to monitor. The viewer's CSV path
(`scan.py:162`) uses `float(...)` instead — the two readers disagree. Coerce
explicitly.

### M13. EIG-vs-random comparison couples the two arms' response noise *(statistical confound)*
`src/subjective_randomness/adaptive_recovery.py:568-570, 667-683`

Both the `eig` and `random` arms build `rng = np.random.default_rng(seed + repeat)`
and draw `rng.binomial(...)` over index sets of equal length, so they consume the
*same* uniform stream — the k-th stimulus in each arm gets correlated Bernoulli
noise. In `compare_model_recovery` the response RNG depends only on `repeat`, so
every generating model and arm in a repeat shares one stream. This is neither
independent noise nor a clean common-random-numbers design (the arms have
different stimuli), so the reported r/RMSE/accuracy differences are partly an
artifact of the coupling. Fold `arm` and `gen_index` into the seed.

### M14. `model_logp_is_finite`-style silent model dropping in predictions
`src/models/theorist/predictions.py:44-49`

`get_model_predictions` catches `(KeyError, FileNotFoundError, ValueError)` and
`continue`s, so a model that fails to import or raises inside its prediction logic
silently vanishes from the result dict (skewing any downstream weighting/EIG),
while `TypeError`/`AttributeError` are *not* caught — inconsistent. Log dropped
models and their reason.

### M15. `datetime.utcnow()` participant IDs can collide within a second
`src/pipelines/outer_loop/collect.py:657,890`

`batch_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")` has 1-second resolution
and feeds participant IDs (`{project}_run{run}_{batch}_p{i}`). Two collection
passes in the same second produce identical IDs; the Firebase results filter keys
on `participant_id_str`, so collisions cross-attribute responses. (`utcnow()` is
also deprecated and naive.)

---

## LOW severity

### L1. `periodicity_score` has a discontinuity at 0.5 and a truncated period range
`src/subjective_randomness/features.py:156-167` (duplicated in `model_families/common.py:163-179`).
Seeds `best_match = 0.5` (any match rate below 50% → score 0) and only tests
`period in range(1, n//2+1)`, excluding near-full-period structure. This feature
feeds the PyMC models, so the asymmetry can distort what they can learn. Also: the
two copies will drift.

### L2. `model_complexity` doc says "chars", code counts "lines"
`src/model_comparison/posterior.py:50-68` vs module docstring (line 11) and CLI
help (line 247). The Occam constant `-0.05` is tuned for *line* counts; a user
trusting the docs would mis-scale the penalty by orders of magnitude. Splitting on
`#` (line 65) also miscounts `#` inside string literals.

### L3. Two-sided empirical p-value `p_value_is_floor` diagnostic is wrong at ties
`src/critique/ppc.py:194-214`. Because `n_ge` uses `>=` and `n_le` uses `<=`, a
replicate equal to `t_obs` is double-counted, so a genuine Monte-Carlo floor case
(`t_obs` at the most-extreme replicate) can report `p_value_is_floor=False`.

### L4. Re-`exec` of each test statistic per replicate
`src/critique/ppc.py:167-168`. The statistic source is `exec`'d once per
replicate (≈201× per statistic × 8). Wasteful, and any module-level side effects
in agent code re-run each time. Compile once, call N times.

### L5. `_thin_posterior` keeps the first draws of each chain and under-counts
`src/models/pymc_inference.py:462-467`. `per_chain = max(1, max_draws // n_chains)`
floors below `max_draws` and keeps each chain's *earliest* draws rather than
thinning by stride. Only affects the posterior-mean `predict_p_left`, so impact is
small.

### L6. Pydantic / non-`ValueError` exceptions escape the viewer's error handler
`src/viewer/scan.py:206` (`TrajectoryStep(**step)` → `pydantic.ValidationError`,
not a `ValueError` in pydantic v2), `scan.py:238,249` (`max(posteriors, key=...)`
→ `TypeError` on `None`/string values). The `@app.errorhandler(ValueError)` won't
catch these, so corrupt `history.json` / `model_posterior.json` yields an opaque
500 (with a traceback in `debug=True`) instead of the loud, file-named error the
module promises.

### L7. Design validator's degenerate-design check has no meaningful floor
`src/validation/stages/design.py:51-56`. Rejects only when `max(eig_values) <= 0`;
an all-tiny-but-positive EIG design (e.g. `1e-9`) passes as "valid", masking a
degenerate design.

### L8. Swallowed exceptions hide failures
`src/validation/validators.py:64-73` (`except Exception: pass` around the
validation-failure logger — the audit trail can silently vanish);
`src/experiments/references.py:44-54` (`except Exception: continue` drops corrupt
PDFs silently and the final truncated chunk near the char cap is dropped);
`src/pipelines/outer_loop/deployment/firebase.py:128-138` (malformed `.firebaserc`
→ `None` silently). Log instead of swallowing.

### L9. Non-reproducible synthetic data via unseeded global `random`
`src/pipelines/outer_loop/collect.py:1311,1324,1327`. `_generate_from_models`
uses the unseeded global `random` module, unlike `_generate_from_pymc_models`
(which seeds `random.Random(seed)`). Synthetic ground-truth data is then
non-reproducible.

### L10. Minor robustness gaps
`.secrets` parsing keeps surrounding quotes (`llm.py:89-106`, `prolific.py:54-67`).
`build_command` falls through to `None` for an in-`_DEFAULT_MODEL` backend with no
branch (`coding_agent.py:43-72`) → `Popen(None)` later. `parse_problem_definition`
silently yields an empty range for `8-4` and swallows malformed ranges
(`experiments/problem_definition.py:46-52`). `random` control set in
`discriminating_probe.py:158-160` is not disjoint from the adversarial set.
`parse_rating` truncates a float rating with `int(value)` instead of rounding
(`model_similarity_judge.py:136-138`).

---

## Documentation & repo hygiene

### D1. README "Project Layout" is substantially stale *(verified)*
`README.md:217-285`. It documents an `inner_loop/` with
`core.py`/`likelihood.py`/`fitting.py`/`bmc.py`/`zoo.py`/`orchestrator.py` — none
of which exist (the real files are `pymc_orchestrator.py` + `run.py`). It lists a
top-level `legacy/` tree (`run_pipeline.py`, `cloudrun/`, `remote_jobs/`, …) that
**does not exist** in the repo, omits `runtime/coding_agent.py`, and omits the
entire `src/subjective_randomness/`, `src/model_comparison/`, and `src/critique/`
trees. `CLEANUP_PLAN.md` likewise references `legacy/`, `src/eig/`, `src/stats/`,
`src/agents` that are absent.

### D2. Default coding-agent backend: code says opencode, docs say claude *(verified)*
`src/runtime/coding_agent.py:25` sets `DEFAULT_BACKEND = "opencode"`, but
`README.md:90` ("The default is Claude Code") and `run.py:491-492` ("Defaults to
the CODING_AGENT env var, then 'claude'") both claim Claude. A user who omits
`--coding-agent` and `CODING_AGENT` gets opencode/Gemini, not Claude. The opencode
default model (`google/gemini-3.1-pro-preview`, line 28) also contradicts
`README.md:101` ("`anthropic/claude-sonnet-4-6`"). Pick one default and make code
+ docs agree.

### D3. Large/loose/duplicated files committed *(verified)*
- `gt.txt` (≈21 MB) is tracked at the repo root.
- Loose one-off scripts at the root: `candidate.py`, `extract_pdf.py`,
  `generate_candidates.py`; a `scratch/` dir of `*.sbatch` + `test_*.py`.
- The project tree is **duplicated**: `projects/<project>/...` at the root *and*
  `src/pipelines/outer_loop/projects/<project>/...`, including the same reference
  PDFs committed twice. Likewise `src/subjective_randomness/model_families/` vs
  `.../pymc_model_families/` vs `.../seed_models/` hold parallel copies of the
  four model families that can drift apart.

Consider `.gitignore`/Git-LFS for the large artifact, removing the scratch
scripts, and collapsing the duplicated project trees to a single source of truth.

(Good: `.secrets` itself is **not** tracked — only `.secrets.example`.)

---

## Verified as NOT problems (so they aren't re-investigated)

- Viewer path traversal (`server.py::_run_dir`) is correctly blocked via
  `.resolve()` + containment check; `send_from_directory` blocks `../`.
- No shell-injection in subprocess calls — all use list-form argv with
  `shell=False`; the agent prompt is a single argv element; URLs use
  `urllib.parse.quote`. No secrets injected into argv.
- The softmax in `model_posterior` subtracts the max before `exp` (numerically
  stable); the `+1` Laplace correction on empirical p-values is the right
  convention; log-domain posterior updates in the recovery code guard underflow
  and clip probabilities.
- `observed_response_data` fails loudly on 0 or >1 observed RVs; `_seed_model_set`
  / `_admit_candidate` fail loudly on missing files; sequential Bayesian model
  evidence in `adaptive_recovery` is computed correctly (dropping the binomial
  coefficient is fine — it cancels across models).
- `local.py:115` `plan` is always bound when reached (`live` implies `!= none`).
</content>
