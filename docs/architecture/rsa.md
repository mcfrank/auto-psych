# RSA reference games — `src/rsa/`

Moved verbatim from `CLAUDE.md` (2026-10-28); `CLAUDE.md` keeps the rules that span files.


Rational-speech-act models of reference games (Frank & Goodman, 2012). A
**parallel stack** (user decision: keep it separate from the TaskSpec
pipeline) reusing the shared runtime, Prolific, Firebase functions and
`src/critique/ppc.py`. **Start at `docs/auto_rsa/CAMPAIGN_LOG.md`** (every
decision and run); `LOOP_ALGORITHM.md` spells out the live loop.

- Models are memo files (`src/rsa/model_file.py`) fitted with numpyro NUTS
  (`src/rsa/fit.py`); selection by grouped cross-validation; candidate agents
  have **no network** (`src/rsa/loop/no_network.py`); self-check
  `python -m src.rsa.loop.check_candidate`.
- **Data**: schema and coding decisions in
  `src/pipelines/outer_loop/projects/rsa_reference/data/README.md`, licences in
  `docs/auto_rsa/DATASETS.md`. **Never commit trial-level data from an
  unlicensed or NC-ND source** (aggregates are fine); those are fetched by
  `src.rsa.ingest` into gitignored `data/rsa/external/`.
- **The pre-registered final experiment** (`src/rsa/prereg/`,
  `docs/auto_rsa/PREREG_FINAL.md`, OSF https://osf.io/s48jv, tag
  `rsa-prereg-final`). **Never change the frozen files**
  (`data/rsa/prereg/design/`, `data/rsa/live/finals_pool_preds_final.json`,
  `data/rsa/live/bar_pool_preds.json`, `src/rsa/prereg/analysis.py`; sha256s on
  OSF) **and don't move `src/rsa/prereg/`** (the bundle hashes code by path).
  Its data are spent: nothing chosen after seeing them may be judged on them
  again (`docs/auto_rsa/CONSOLIDATION.md` §5).
- `data/rsa/` holds run records: never Prolific IDs or trial-level live data.
- Cluster scripts: `scripts/rsa/slurm/`, live launchers `scripts/rsa/live/`.

