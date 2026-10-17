# Merging `number-game` (and `auto-rsa`) into `main`

Written 2026-10-16 from an examination of the branches; nothing has been
merged yet. Read all of it before running anything.

## 0. Status (2026-10-16, later): the scrub replay is done

`number-game` has been replayed onto `771e7eba`, the rewritten twin of its old
base: `git rebase --onto 771e7eba 7135d6d8 number-game`. It no longer contains
`7135d6d8`, and no commit reachable from it holds a raw Prolific ID, so it is
safe to push. The pre-replay branch is kept, local only, as
`number-game-prerewrite-backup`: never push it. The replay onto current
`origin/main` (§ 3, conflicts § 4) is still to do; start it from
`771e7eba..number-game` instead of `7135d6d8..number-game`.

Old → new hashes (the dataset run in `/scr/wurgaft/auto-psych/number_game_dataset`
recorded `573e2c9a`, then `2395ef69`, as its `code_commit`):

| old | new | commit |
|---|---|---|
| `8419479d` | `92f85a22` | Number game: PyMC seed port of Bigelow's numpyro rules + intervals model |
| `ae4a3770` | `92f865ff` | Number game: Tenenbaum (2000) seed; B&P seed renamed |
| `403f3acf` | `80d51b2d` | Number game: fix the Tenenbaum seed's example noise at 0.1 |
| `b1246bb3` | `c19da24b` | Generalise the pipeline to a per-project task; number-game recovery runs |
| `912746e0` | `e4674d6f` | Grouped cross-validation as an inner-loop selection criterion |
| `4cf9d9a2` | `c92ddb75` | hide_participant_id: population-level models only |
| `3ef63e0a` | `44c9f0fb` | No PSIS-LOO under grouped CV |
| `f9fd5144` | `7bcbc3f6` | Number game: a third seed, rules + intervals with a midpoint prior |
| `c7a48c0d` | `2b68e117` | fit.method: warm_start_nuts, NUTS started from the posterior mode |
| `a3a3b4cf` | `927d4479` | Faster CV fits: folds concurrently, fewer compiles, memoised held-out densities |
| `ab8a3a63` | `75cbe421` | Number-game recovery runs on the SC cluster (CLUSTER=coco) |
| `b9e42c10` | `fd3e8440` | archive_agent_tree: read the archive back without grep -q |
| `e1912acc` | `07170c54` | Number game: every seed is a recovery ground truth |
| `d10aa1d2` | `ab767574` | Agent sandbox: mount the resolver config's target (DNS on systemd hosts) |
| `50246c80` | `d4e9e2e7` | Agent sandbox: python/python3 are the harness's venv |
| `5feacbc1` | `01b4059f` | CV screen: round each fold's share of the fit workers up |
| `fd380192` | `d6dbcaff` | Compile directories kept across jobs on coco (AUTO_PSYCH_COMPILE_SLOTS) |
| `344056f2` | `5afb8cfd` | Number game: dataset runs (data_source: dataset) |
| `7f916aa4` | `4ce04545` | Inner loop: overlap fitting with agent time |
| `80cd4d97` | `287a4001` | Docs: the number game and the SC cluster, briefly |
| `ecd67e27` | `ff35cea6` | Agents compile in kept, per-cell compile dirs |
| `4d7aa5d5` | `ab6ddcfb` | Candidate self-check fits with the run's fit method |
| `f923686c` | `46ec15d4` | Claude billing mode subscription_then_api |
| `d9dfc615` | `5dd5f2ca` | Number game: the dataset run's config and job support |
| `573e2c9a` | `e86a7f02` | Number game: submit_dataset_run.sh for dataset runs |
| `7bfe5f8f` | `168627e8` | subscription_then_api returns to the subscription at the limit's stated reset |
| `2395ef69` | `1f0f7ec7` | Sampling warnings are computed once per fit, repeated on cache hits |
| `375a82c6` | `6f3df266` | R-hat over the free parameters only; drop the per-fit warning cache |
| `c5217946` | `d82cbab9` | Dataset runs: summarise by best model, chain no test-retest analysis |
| `5c0cfe95` | `7eff7850` | Cleanup backlog: audit of removable code (2026-10-16), to do after merging main |

## 1. The history problem — read first

`origin/main`'s history was **rewritten on 2026-10-01** to scrub raw Prolific
IDs from `data/results/human_experiment/completion_times.csv`
(`docs/onboarding/full_run_checklist.md`, "Commit hashes recorded before the
2026-10-01 history rewrite"). Every commit got a new hash.

`number-game` (local only, never pushed) is built on the **old, unscrubbed
history**, and so is the local `main` (`7135d6d8`).

- Its fork point is old `7135d6d8` ("Novelty gate: predict each model on the
  pool once per model loop"); the same commit on the rewritten history is
  **`771e7eba`**. Their trees differ only in that CSV and one test line.
- `git merge-base number-game origin/main` is the repo's first commit — do
  **not** `git merge` the branches as they are: that would pull the whole old
  history, raw IDs included, into main.
- **Never push** `number-game`, local `main`, or any branch whose ancestry
  includes `7135d6d8`. Check before every push:
  `git merge-base --is-ancestor 7135d6d8 <branch> && echo OLD-HISTORY-DO-NOT-PUSH`.

The fix is a replay: `number-game`'s own commits (29 linear commits as of
2026-10-16, `7135d6d8..number-game`, all by DanielWurgaft, no merges) are
rebased onto the new history.

## 2. Order

`origin/auto-rsa` (Mike Frank's, 162 commits) already contains all of
`origin/main` and was prepared for merging (its commit `723b8962`, "Docs for
merging auto-rsa into main"); into main it is a fast-forward. **Its merge is its
owners' call** — confirm with the user before touching it. Two things for the
team to agree on first: it moves the repo to Python 3.12 (§ 5), and it adds
~235 MB / ~5,800 files under `data/rsa/` to main.

- If `auto-rsa` lands first (preferred): replay `number-game` onto the new
  `origin/main` (= `auto-rsa`). Three conflicts.
- If not: replay onto `origin/main`. Two conflicts (no `.gitignore` one).

Do not start while a number-game experiment is running from this checkout —
ask the user.

## 3. Steps

```bash
cd /sailhome/wurgaft/auto-psych
git status                       # must be clean (commit or ask about anything)
git fetch origin
git branch number-game-prerewrite number-game   # keep the old branch, local only
git log --reverse --format='%h %s' 7135d6d8..number-game > $TMPDIR/old_hashes.txt
git checkout -b number-game-rebased number-game
git rebase --onto origin/main 7135d6d8          # origin/main after auto-rsa lands,
                                                # or origin/auto-rsa if replaying onto it
# resolve conflicts (§ 4), `git rebase --continue` each time
git merge-base --is-ancestor 7135d6d8 HEAD && echo "STILL OLD HISTORY" # must print nothing
git log --reverse --format='%h %s' origin/main..HEAD > $TMPDIR/new_hashes.txt
```

Then:

1. **Record the old→new hash map** (pair `old_hashes.txt` and `new_hashes.txt`
   by subject) in a short doc next to the number-game results, the way the
   checklist did for main. Finished number-game runs recorded old hashes
   (`code_commit`, `code_provenance.json`, `holdout.json`); the map keeps them
   traceable. Ask the user where they want it.
2. Apply the extra fix in § 4.4 as its own commit.
3. Run the tests (§ 6).
4. Point local `main` at the new history: `git checkout main && git reset --hard
   origin/main` (only on a clean tree; local `main` has nothing that is not on
   `origin/main`).
5. Push `number-game-rebased` (after the ancestry check above) and open a PR
   into `main` — only when the user says so. Keep `number-game-prerewrite`
   local until the user has archived their results.

## 4. Conflicts and their resolutions

All are mechanical. Found with `git merge-tree --write-tree
--merge-base=7135d6d8 <target> number-game`; a rebase hits them commit by
commit, mostly in `b1246bb3` ("Generalise the pipeline to a per-project task").

### 4.1 `src/pipelines/outer_loop/eig.py`

`number-game` threads a `task` (a `src/tasks` `TaskSpec`) through the design
code; main's `a93563b7` ("Design over person-level models as new participants")
added a `responses_csv=` keyword so a model with a participant effect is
predicted as new participants (`_new_participant_ids`, `_new_participant_draws`).
**Keep both.**

- `_screen_usable_models(model_names, models_dir, rows, task, *, responses_csv=None)`.
- In the per-model prediction: keep main's `try / except MissingStimulusColumns
  → _new_participant_draws(...)` path, and pass
  `var_name=task.prediction_var` in **both** branches. The new-participant path
  (`_new_participant_draws`) must also predict `task.prediction_var` (not a
  hard-coded `p_left`); thread `task` into it. Git cannot see this one.
- The call site: `_screen_usable_models(model_names, models_dir, rows, task,
  responses_csv=responses_csv)`, keeping main's comment.
- Docstring: number-game's wording ("prediction draws … prediction is
  defined") plus main's sentence about participant effects.

### 4.2 `tests/test_design_inputs.py`

Three monkeypatched stubs of `_screen_usable_models`; both sides changed the
signature. Use `lambda names, d, rows, task, **k: (names, [])`.

### 4.3 `.gitignore` (only when replaying onto `auto-rsa`)

Both sides appended lines at the same place. Keep both: auto-rsa's `data/rsa/…`
block and number-game's `data/number-game/`.

### 4.4 Not a conflict, but needed: the RSA agent-tree exclude (only with `auto-rsa`)

`number-game` added `/src/number_game/` to
`scripts/subjective_randomness/slurm/agent_tree.exclude`. auto-rsa's
`tests/test_rsa_slurm_scripts.py::test_every_subjective_randomness_rule_carries_over`
requires every rule there to be in `scripts/rsa/slurm/agent_tree.exclude` too,
and fails on the merged tree (`src/number_game … is not excluded`). Add
`/src/number_game/` to `scripts/rsa/slurm/agent_tree.exclude`.

### 4.5 Auto-merged, read through

Both sides edited these and git merged them cleanly; check they still read as
one document / agree: `CLAUDE.md` (both add sections — number game, RSA),
`README.md`, `scripts/subjective_randomness/slurm/{_env.sh, README.md,
holdout_setup.sbatch, holdout_recovery_array.sbatch}`,
`scripts/outer_loop_live/_pilot_config.py`, `src/models/data_binding.py`,
`src/pipelines/inner_loop/pymc_orchestrator.py`, `src/runtime/agent_sandbox.py`,
`src/runtime/coding_agent.py`.

Semantic overlaps worth a look (main changed behaviour number-game's code may
assume otherwise): main's pruning at `4·dse` and the re-prune tool
(`a93563b7`) vs number-game's grouped-CV selection (`inner_loop.selection`,
no PSIS-LOO under CV); main's new-participant design path vs
`hide_participant_id`.

## 5. Python 3.11 → 3.12

`auto-rsa` sets `.python-version` to 3.12 (memo, its model language, needs it)
and keeps the arviz 0.x / pymc 5.x API by capping `arviz>=0.17.0,<1`,
`pymc>=5.10.0,<6` in `pyproject.toml`; it also adds an `rsa` dependency group
(memo-lang, jax/jaxlib 0.7.0, numpyro 0.19.0, all marked `python_version >=
'3.12'`) to `default-groups`. **Keep those caps** — the 3.11 pin only existed
because uncapped 3.12 resolves arviz 1.x / pymc 6.x.

**Verdict: 3.12 with the caps breaks nothing.** Checked 2026-10-16 on
exported copies (git archive, not checkouts) under `/scr/wurgaft/py312_check/`
(logs in `logs/`, venvs in `venvs/`):

- **Same libraries.** With the caps, 3.12 resolves exactly the versions of
  number-game's 3.11 lock: pymc 5.28.5, pytensor 2.38.3, arviz 0.23.4,
  numpy 2.2.6, scipy 1.14.1. Without them: pymc 6.0.1, pytensor 3.0.4,
  arviz 1.1.0.
- **Same numbers.** A seeded fit + PSIS-LOO of two seed models gave
  bit-identical ELPD-LOO and posterior means on 3.11 and capped 3.12, so
  finished 3.11 runs stay comparable.

Fast suite (`-m "not slow"`):

| config | fast suite |
|---|---|
| `number-game`, Python 3.11 (baseline) | 22 failed / 1634 passed |
| `number-game`, 3.12 **with** the caps | 22 failed / 1634 passed — the same 22 |
| `number-game`, 3.12 **without** caps | 72 failed (arviz 1.x / pymc 6.x API) |
| merged tree (`number-game` replayed onto `auto-rsa`), 3.12 | 49 failed first run; re-run of those: 48 pass, 1 fails (§ 4.4) |

The 22 baseline failures are the test environment, not code: re-run alone,
21 pass; the last needs `PROLIFIC_API_TOKEN` (absent from the copy). Some
others read git, which an exported copy lacks. The 48 that passed on re-run
were load-related flakiness (a real experiment was running on the node).
The slow (MCMC) suite was not completed: it was stopped about 20 tests in, all
passing so far — run it on the replayed branch (§ 6).

**Needed with the merge (if `auto-rsa` is in):** number-game's
`scripts/subjective_randomness/slurm/_env.sh` hard-codes the coco venv's
Python — `if [[ "$CLUSTER" == coco ]]; then export UV_PYTHON="${UV_PYTHON:-3.11}"`
— while its comment says coco uses the repo's `.python-version`. Change the
coco default to 3.12 (Sherlock's branch already defaults to 3.12, and
auto-rsa's own `_env.sh` sets 3.12), or the coco setup jobs keep building 3.11
venvs against a 3.12 project. Then rebuild the coco venvs (they live under
`/scr/wurgaft/...`). The setup `.sbatch` files take `$UV_PYTHON` and need no
change.

## 6. Tests

On the replayed branch, Python 3.12 if `auto-rsa` is in:

```bash
uv sync
uv run pytest -q -m "not slow"     # fast suite
uv run pytest -q                    # full suite once, before pushing
```

Expect zero failures in the real checkout; anything that fails should be
explained against § 5's table before the PR.

## 7. Decided: the RSA loop stays separate

auto-rsa's loop (`src/rsa/`, memo/JAX + numpyro, categorical choices among
objects) is a deliberate parallel stack (`docs/auto_rsa/PLAN.md`, decision of
2026-10-06); number-game's `src/tasks` `TaskSpec` generalises the shared PyMC
pipeline for binary responses. The user decided (2026-10-16) **not** to unify
them. The merge must not try to: no backend seam, no moving RSA onto
`TaskSpec`. Optional, separate from the merge: deduplicating leaf code, e.g.
one grouped-CV module (`src/model_comparison/cv.py` and `src/rsa/loop/cv.py`).
