# Inner loop — `src/pipelines/inner_loop/`

Moved verbatim from `CLAUDE.md` (2026-10-28); `CLAUDE.md` keeps the rules that span files.


`pymc_orchestrator.py` orchestrates; `model_zoo.py` does seeding, admission,
pruning and the novelty gate; `scoring.py` scoring and export;
`candidate_agent.py` briefs and spawns agents. Each round: optional critique →
candidate agents in parallel → sequential admission.

- **Starting models are not protected** (user decision 2026-09-28): pruned,
  capped and dropped like any model. They are recorded once in
  `<run>/starting_models.json` (`{"starting_models": [...],
  "starting_models_prunable": true}`); a bare-list record means a run started
  with protected seeds and refuses to continue (the conditions must not mix).
  No candidate may reuse a starting model's name or that of a model retired
  earlier in the run (`reserved_names`); clashes are renamed `<name>_2`.
- **Admission** (`_admit_candidate`): a loadable `candidate.py` (module-level
  `model: pm.Model`) plus `hypothesis.md`, passing in order: the **code gate**
  (`import_gate.py`: import allowlist, no file access or reflection escapes —
  agent code runs in the harness process); the logp gate; the **data contract**
  (`src/models/model_contract.py`: the likelihood must be Bernoulli(chose_left;
  p_left) over the CSV's rows, no `pm.Potential` on the responses; a starting
  model that breaks it raises); the fit and finite-ELPD gates; the
  **convergence gate** (≤0.1% divergent, R-hat ≤ 1.05, bulk ESS ≥ 100; a near
  miss is refit once at `target_accept` 0.95; a model's declared
  `target_accept` is a floor, so rejections advise reparameterising); the
  **time limit** (`CANDIDATE_FIT_TIME_LIMIT_SEC` = 30 min per sampling run,
  user decision 2026-09-28; starting and carried models are never limited); and
  the **novelty gate** (posterior-mean `p_left` ≥ `novelty_rmse_threshold`,
  0.002, RMSE from every admitted model on the loop's own 512-stimulus novelty
  pool — deliberately not the training stimuli nor the recovery harness's eval
  pool). A candidate binding `trial_index`, or undefined or failing on the
  pool, is rejected. A code error raised in the candidate's own file is a
  rejection; the same error from the harness, or an infrastructure error,
  raises (`is_model_failure`).
- **Retry and repair** (`_Slot`): a slot whose agent wrote nothing is
  re-spawned once (`candidate_<i>_retry_1/`); a rejected candidate is
  re-spawned once with the rejection verbatim (`candidate_<i>_repair_1/`); a
  repair is final. Every attempt is a ledger line. An agent stopped by a usage
  or rate limit (`src/runtime/usage_limits.py`) waits for the stated reset and
  reruns, up to 12 h (`AgentUsageLimitExceeded`); a login that cannot pay
  raises `AgentLoginFailed`. Agents self-check with `check_candidate.py`.
- **Slot roles** (`slot_roles`): with ≥ 4 slots, `C - 3` exploratory slots walk
  the task's lens battery (`TaskSpec.candidate_hints`), two **refine the
  incumbent** and one **refines a model of the agent's choosing** from
  `refinement_menu.md` (live and pruned models with hypotheses and sources).
  Refinement briefs lift the anti-grafting and no-composition clauses. The role
  is the slot's assignment, recorded in the ledger context; which model an
  agent refined is prose in its `hypothesis.md` and is **never parsed**. The
  holdout configs run five rounds of six per experiment.
- **Pruning** (`_prune_losers`) runs **once, at the end of each experiment**:
  trusted models (reliable PSIS-LOO and converged) with `elpd_diff >
  dse_multiplier · dse_clustered` behind the best trusted model move to
  `models/pruned/` (`dse_clustered`, `src/models/clustered_se.py`: the
  trial-level `dse` is ~2x too small). Then `_cap_live_set` keeps at most 8
  (untrusted first, then lowest ELPD). The best trusted model always survives.
  No stacking-weight floor (stacking weights are ensemble coefficients). The
  step is `pymc_orchestrator.end_experiment`; `outer_loop/reprune.py` redoes it
  for a finished experiment at another multiplier without MCMC. Recovery sweeps
  prune at 2, the October 2026 live series at 4 (user decision 2026-09-30).
- **Ledger** (`hypothesis_ledger.py`): `model_loop/attempted_hypotheses.jsonl`
  records every slot and prune, carries across experiments, and is rendered
  into every brief as "already tried — do not re-propose". A pruned mechanism
  may return with a substantive change. **`LedgerEntry` must not gain
  fields**: `from_json` requires an exact key-set match, so a new field makes
  every inherited ledger unreadable.
- **Export / "best"**: `_best_exportable_model` (`scoring.py`) — the best model
  by ELPD-LOO rank among those with reliable PSIS-LOO, **never the softmax
  posterior argmax** (rounded, so far-behind models tie at 0.0). The same rule
  gives `history.json`'s `best_model`, the critique incumbent and `SUMMARY.md`.
  `cognitive_models/` becomes the **live set**: every survivor, not only the
  winner. "Reliable" is `src/models/loo_reliability.py`'s verdict, not arviz's
  blanket k > 0.7 flag; constant-log-likelihood trials are exempt and scored
  as exactly constant (rounding noise there makes PSIS return NaN).
- **Registry** (`src/registry/io.py`): after each experiment a **uniform prior
  over the carried set**, which the next design reads. Not stacking weights:
  those made every present model's prior ~0 (zero EIG everywhere).

