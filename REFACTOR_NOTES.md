# Branch `fix-tests-and-refactor` — what changed and why

**Date:** 2026-08-08. **Baseline:** `7936d8a` (merge of PR #10, km-stim-complexity). **Result:** full suite green — 939 passed, 3 skipped, 0 failures (was 841 passed, 3 failures).

This branch fixes the three failing tests, then works through a reviewed backlog of refactors in eight commits, one per phase. Two commits deliberately revise parts of PR #10's conflict resolution; those are called out below so the reasoning is visible to everyone who worked on that PR.

## Why tests were failing

**1. The PR #10 merge silently reverted the minimal-DP `parse_motifs` fix.**
Commit `8667585` (main) fixed `parse_motifs` to the correct Falk & Konold minimal-description parse (DP over partitions into constant-run chunks costing 1 and strictly-alternating chunks costing 2, ties broken toward fewest chunks) in *both* `src/subjective_randomness/features.py` and `src/subjective_randomness/model_families/common.py`. The conflict resolution in `bb91a42` kept the km-stim-complexity branch's older greedy run-based parse in `common.py`, while `features.py` kept the DP version. The two implementations disagree on 64 of the 510 sequences of length ≤ 8 (first at `HTHHT`: DP parses `HTH|HT` → (0, 2); greedy parses `HTH|H|T` → (2, 1)). This broke:

- `test_parse_motifs_implementations_agree_exhaustively` (direct comparison), and
- `test_new_models_match_their_pure_python_twin[falk_konold_dp]` — the PyMC model scores DP-parse feature columns from `features.py`, while its pure-Python twin called the regressed `common.parse_motifs`.

**2. The legacy-design golden test was machine-dependent and never passed here.**
`test_legacy_compat_reproduces_the_historical_golden_output` pinned a byte-exact 5-pair selection. Checked out at the commit that introduced it (`1c8436c`, pre-merge, same numpy 2.2.6), it already failed on this machine. The greedy selection ranks candidates by a Monte Carlo EIG estimate whose exact bits depend on libm's `log`/`exp` rounding, and the lengths-(3,4) universe is full of complement-symmetric pairs whose true EIGs are exactly equal — so which member of a tie gets picked differs across platforms on last-ulp differences, with no numerical error anywhere. The pin encoded the generating machine's tie-breaking. (There is no CI, so this was never caught.) A secondary issue: the conflict resolution pinned `_LEGACY_COMPAT_MODEL_FAMILIES` in non-sorted order, whereas the historical algorithm enumerated the family package with `pkgutil` and sorted it — model order is load-bearing (it permutes the float summation order and the sampled scenario identities in the greedy loop).

## The eight commits

### `16d5659` — restore the minimal-DP motif parse *(revises PR #10's conflict resolution)*
- `common._parse_motifs` carries the DP algorithm again (clean-once + `lru_cache` wrapper kept).
- `sequence_stats._run_and_motif_stats` — a third, vectorized copy of the parse used by the fast exhaustive-design pipeline's quotient — brought onto the same DP semantics (bit-exactness test retained).
- Pinned expected values that legitimately shift with the parse were regenerated: the quotient class-count goldens in `test_sequence_stats.py`; the greedy reference in `test_model_families_common.py` was replaced with an independent brute-force minimal-partition reference (all sequences ≤ 10). The Falk & Konold worked-example pins (DP 4.0/5.0/7.0) are identical under both parses and unchanged.

### `066501f` — machine-stable pin for the legacy design algorithm *(revises PR #10's golden-pin design)*
- `_LEGACY_COMPAT_MODEL_FAMILIES` restored to sorted order (the historical referent).
- The golden test now pins what is stable across machines: the EIG multiset of the historical selection (`{0.120592, 0.125002 ×2, 0.149885 ×2}`), the tie-free first pick `("HTH", "HTTH")` (its margin over the runner-up is ~4.6e-3 relative — no rounding difference can cross it), selection bookkeeping, the candidate universe, and in-process determinism. The docstring documents why a byte-exact cross-machine pin is impossible for this pipeline.

### `c6245aa` — one seed-model source of truth: the faithful four
The repo had drifted into two fully disjoint "active" model sets: the outer loop seeded experiments from `seed_models/` (the four hero-run replicate winners, promoted 2026-07) while EIG design and recovery used `pymc_model_families/models_manifest.yaml` (the four literature-faithful models, consolidated 2026-08). Both manifests claimed to be the source of truth, and `model_recovery.default_generating_params` raised `ModuleNotFoundError` for every hero name (no pure-Python twins). **Decision (Ben, 2026-08-08): faithful four everywhere.**
- The seed pool now mirrors the registry manifest (a test asserts pool == registry).
- The hero winners are preserved at `seed_models/archive_hero_run_2026_07/` with a README on provenance and how to resurrect them.
- All ~13 hand-rolled manifest parses go through one loud reader: `src/models/model_manifest.py` (+ `pymc_model_families.REGISTRY_DIR`).
- A drift-landmine test asserts every manifest name resolves to a loadable PyMC file *and* a pure-Python twin.
- **Consequence to be aware of:** `holdout_recovery_faithful` / `_deepseek` configs now perform a real holdout — the ground truth is in the seed pool, so experiment 1 seeds from the other three faithful models rather than four unrelated hero models. Correct semantics, but it changes what those configs measure relative to the 2026-08-06 runs.

### `79d52e4` — dedup core helpers
- Featurizer single-sourcing: `common.py` delegates `occurrence_probability`, `periodicity_score`, `local_imbalance` (and the motif parse) to `features.py`; it keeps only its clean + cache wrapper layer. The empty-input behavioral fork (features returned 0.0; family twins raised) is resolved toward **raising** everywhere — verified no legitimate producer of empty sequences exists (all 105 response CSVs in `data/` checked).
- One `load_featurizer` (`src/pipelines/outer_loop/featurizer.py`) on the strictest policy — a featurizer file that exists but lacks `featurize_stimulus` now raises instead of silently yielding unfeaturized rows.
- One repo root: `src.runtime.config.REPO_ROOT = pyprojroot.here()`; ten hand-rolled `parents[N]` variants removed.
- `argparse` → `tyro` (repo convention): `evaluate_recovery.py`, `scripts/smoke_open_participant.py`, `analysis/behavioral/fit_mega_models.py`.
- Deliberately *not* merged: `theorist/predictions._normalize_stimulus` vs `common.normalize_stimulus` (different contracts, and merging would invert the layering between the project-agnostic model layer and `subjective_randomness`).

### `12071c9` — fail-loud sweep
Policy: an unexpected failure raises; an expected-and-handled outcome has an explicit, logged policy. Highlights:
- `collect.py`: the double `except Exception: pass` in the participant driver (`_click_random_choice`/`_act_key`) — the exact masking behind the historical degenerate all-left-data incident — now logs the button→keyboard fallback and raises if both modalities fail; programming errors propagate. `get_llm()` failures raise instead of silently degrading every simulated participant to random clicking. Live collection raises on missing `prolific_study_id`/`results_api_url` or failed `/results` fetches instead of returning `[]` ("no participants" is no longer conflated with "collection broken").
- `theorist/predictions.py` raises when a model fails to load/predict (no more silent renormalization over fewer models); `eig._screen_usable_models` still screens numerically unusable models but raises on broken code (`BROKEN_MODEL_CODE_ERRORS`, shared).
- `validators.py` raises on an unknown stage key (was: `validation_ok: True`). `registry/io.py` raises on malformed fields (was: silent coercion). `pymc_inference` diagnostics return `None` + warn when a check *could not be made* (never fabricate 0 divergences). `deployment/manifest.py` raises on git failure (provenance is the point). `runtime/prolific.py` catches are narrowed to `requests.RequestException` with a documented error contract.
- Kept intentionally: inner-loop candidate rejection (agent-written code is expected to fail), PPC NaN rows for agent statistics, `_results_request` retries.

### `504bde3` — dead code removal (−708 lines)
The dead `run_collect` subtree in `collect.py` (superseded by `orchestrator.run_collect_programmatic`), the unused `src/experiments/{problem_definition,references,state}.py`, and seven caller-free functions (`batches_dir`, `run_banner`, `write_transcript`, `create_test_participant`, `stop_usage_log`, `_execute_test_statistic`, `_sha256_dict_arrays`). Every deletion re-verified caller-free by grep at delete time.

### `7ae8744` — test-suite consolidation
- `tests/paths.py` (one `REPO_ROOT`, re-exported from `src.runtime.config`, replacing 46 per-file path constants and 14 copies of the importlib script loader — loads now lazy, so analysis scripts no longer execute at collection time), `tests/recovery_fixtures.py`, `tests/inner_loop_fixtures.py`, `tests/model_registry.py` (model lists derived from the manifest; the one intentional literal pin stays in `test_literature_faithful_pymc.py`).
- The real-NUTS test in `test_eig_pymc.py` is now `slow`-marked; `test_state_loader.py` no longer writes into the actual repo tree (uses `tmp_path`).
- `test_prompt_hygiene.py` no longer crashes on deleted-but-unstaged tracked files.

### `e4c0a74` — repo hygiene (light; no history rewrite)
- Deleted root cruft: `candidate.py` (escaped inner-loop artifact), `generate_candidates.py`, `extract_pdf.py`, `untitled folder/`, stray `__pycache__/`, `scratch/`; stale `CLEANUP_PLAN.md`.
- Untracked but kept on disk: `gt.txt` (21 MB), 99 regenerable MCMC `.nc` caches (~653 MB), `run.pid`. New `.gitignore` rules: `*.pid`, `data/**/mcmc_cache/`. (`.git` stays large — shrinking it needs a coordinated history rewrite, deliberately not done.)
- `projects/` shadow tree unified: checked-in project assets live under `src/pipelines/outer_loop/projects/…` (`config.PROJECT_ASSETS_DIR`); root `projects/` remains the run-output root only. **This uncovered and fixed a real bug: `src/models/project/ground_truth.py` was loading the stale 2-model shadow registry instead of the live 4-model one.** `prolific_config.yaml` moved to the assets dir. The stale shadow copies of `ground_truth_models.py`/`problem_definition.md` are deleted.
- `HERO_RUN_DESIDERATA.md` pruned of resolved audit items (each verified in code first); `PROBLEMS.md` marked as a historical record.

## Known follow-ups (documented, not done)

- Fast-suite wall time is dominated by `test_stimulus_diverse_selection.py` (~66s of ~92s) — the lever if the suite needs speeding up.
- `_server_reachable` / `_start_experiment_server` / `_run_one_participant_browser` in `collect.py` became orphaned when the dead `run_collect` subtree left; they were off the approved deletion list (and two fail-loud tests cover `_server_reachable`) — needs a keep-or-delete decision.
- A transient LLM API error mid-run still degrades that one simulated participant to blind clicking (logged to stderr and `llm_steering_error.txt`) — the last remaining silent-ish degradation.
- The two EIG stacks (pure-Python `stimulus_design.py` vs PyMC `outer_loop/eig.py`) remain separate by design; unifying them is research-risky.
- Optional: git history rewrite to actually shrink the 801 MB `.git` (requires co-author coordination); firebase config consolidation; the dormant `test_models.py` / `test_theorist_output.py` pair.
