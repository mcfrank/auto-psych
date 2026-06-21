# Live outer-loop runs on Sherlock (real Prolific participants)

Runs `src/pipelines/outer_loop/run.py` in **`--mode live`**: Claude Code agents
propose models → design stimuli → implement a jsPsych experiment → it is
**deployed to Firebase Hosting** + Cloud Functions → a **Prolific study** is
created and published → the run polls for human submissions (≤ 2 h/experiment),
fetches results from Firestore, and runs the inner model loop.

Everything runs in a Slurm job (never on the login node). Compute nodes have
outbound internet, so they reach Anthropic (Claude Code), Prolific, Firebase,
and the npm registry directly.

## Files

- `pilot.yaml` — **the one config file you edit.** Project, run label, #experiments,
  design mode, Slurm walltime/qos, the Prolific study (participants/reward/length/
  name), and the modeling settings — all here.
- `run_pilot.sh` — **the launcher.** Reads `pilot.yaml`, renders the project's
  `prolific_config.yaml`, prints a cost summary, asks you to confirm, then submits.
- `_pilot_config.py` — helper used by `run_pilot.sh` (parse/validate/render/cost).
- `_env.sh` — shared env: modules (`gcc/14.2.0`, `nodejs`, `claude-code`, `uv`),
  caches/state off `$HOME`, secrets, CA certs, the deploy lock, threading.
- `setup.sbatch` — **run once per `WORK_ROOT`**: builds the shared uv venv
  (el7-safe wheels), installs `functions/` deps, warms the firebase CLI, and
  preflights the import chain.
- `run_live.sbatch` — the per-run Slurm job (the launcher submits this; you can
  also drive it directly).
- `submit_parallel.sh` — launch `K` parallel runs, each fully isolated.

## Prerequisites (one-time)

1. **Coding-agent auth** — Claude Code is already authenticated under `~/.claude`
   (visible from compute nodes via NFS `$HOME`). Nothing to do unless the login
   expires (`claude` then re-prompts on a login shell).

2. **Secrets** — put these in the repo-root `.secrets` (see `.secrets.example`):
   - `PROLIFIC_API_TOKEN` — Prolific → account settings → API tokens.
   - `FIREBASE_TOKEN` — generate once on a machine with a browser:
     ```bash
     firebase login:ci        # or: npx -y firebase-tools login:ci
     ```
     Paste the printed token as `FIREBASE_TOKEN=...`. This lets `firebase deploy`
     run non-interactively on a compute node.
   - `GOOGLE_API_KEY` — only needed for simulated/Gemini paths; a pure live human
     run does not use it.

3. **Study + run config** — set everything in `scripts/outer_loop_live/pilot.yaml`
   (participants, reward, task length, study name/description, #experiments,
   design mode, walltime). `run_pilot.sh` renders
   `projects/<project>/prolific_config.yaml` from it on launch, so **that file is
   now auto-generated — edit `pilot.yaml`, not it.**

4. **IRB consent** — deployment hard-requires a consent gate; confirm
   `templates/consent.txt` is your IRB-approved wording (it is injected as a
   full-screen "I agree" gate before the experiment).

5. **`.firebaserc`** — already present (default project `auto-psych-2c5da`); the
   scripts also pass `--firebase-project` explicitly.

## Build the environment (once)

```bash
cd ~/auto-psych
sbatch scripts/outer_loop_live/setup.sbatch
# watch: squeue --me   |   then confirm "[setup] live import chain OK" in the log
```

Output venv + caches land under `WORK_ROOT` (default
`$SCRATCH/auto-psych/outer_loop_live`).

## Run a pilot (config-driven — recommended)

1. Edit `scripts/outer_loop_live/pilot.yaml`.
2. Launch:

```bash
cd ~/auto-psych
bash scripts/outer_loop_live/run_pilot.sh            # CONFIG=other.yaml to use a different file
```

It prints a **cost summary** (per experiment and grand total) and the live URL,
asks you to type **`yes`** (this recruits real humans and spends real money),
then submits the job and tells you how to monitor and how to stop. `CONFIRM=yes`
skips the prompt.

**To stop a pilot:** stop/pause the study **in the Prolific dashboard** (that is
what halts recruiting and charges) — `scancel` only stops the pipeline job, not
an already-published study.

> ⚠️ **Multi-experiment caveat.** Only **single-experiment** runs are validated.
> `experiments: >1` with `design_mode: exhaustive` runs the posterior-design path,
> which requires a pure-Python family twin in
> `src/subjective_randomness/model_families/` for **every** model carried into
> experiment ≥2 — including the inner loop's exported `inner_loop_model` and any
> new agent-proposed models, which have **no twin** — so it will raise at the
> experiment-2 design step *after* experiment 1's participants are already paid.
> For a sequence, either use `design_mode: agent` or first do a simulated
> multi-experiment dry-run (below). Keep `experiments: 1` for a safe first pilot.

## Run (advanced: direct / parallel)

**One run, no cost summary/confirmation:**

```bash
sbatch --export=ALL,RUN_LABEL=pilotA,N_PARTICIPANTS=20 \
  scripts/outer_loop_live/run_live.sbatch
```

**Several in parallel** (commit your code first — worktrees check out `HEAD`):

```bash
K=3 N_PARTICIPANTS=20 bash scripts/outer_loop_live/submit_parallel.sh
```

Longer than ~2 days of wall-clock? Set `qos: long` in `pilot.yaml` (or add
`--qos=long`, cap 7 days). CPU only — no GPU.

## How parallel runs stay isolated

| Resource | Isolation |
|---|---|
| Output data tree | `AUTO_PSYCH_OUTPUT_DIR` per run → separate `experiment{N}/` |
| Local checkout (`public/`, `firebase.generated.json`, `opencode.json`, `functions/`) | one **rsync copy** of the repo per run (el7 git is too old for `git worktree`; rsync also needs no commit) |
| opencode session DB / cache | **private XDG data/state/cache + TMPDIR** per run (node-local SSD) |
| Live experiment URL | hosting path `/e{N}-<label>/` (per `--run-label`) |
| Participant data | `collection_sessions/<collection_session_id>/responses` — `collection_session_id` embeds the run-label, so `/results` reads only this run's data |
| Prolific study | each run creates its own study |
| **Firebase deploy op** (single shared site) | serialized by `AUTO_PSYCH_DEPLOY_LOCK` — only the brief deploy step; agents + the ≤2 h poll stay concurrent |

## Validate without recruiting humans

```bash
cd ~/auto-psych && source scripts/outer_loop_live/_env.sh

# Offline dry-run: staging + IRB consent gate + manifest + Prolific payload,
# zero Firebase/Prolific/human calls.
srun -p dev -t 10:00 -c 2 --mem=4G "$VENV_PY" -m src.pipelines.outer_loop.run \
  --project subjective_randomness --experiment 1 \
  --prepare-smoke-experiment --deploy-only \
  --deploy-target dry-run --prolific-mode test --run-label smoke-dry

# Real Firebase deploy smoke (needs FIREBASE_TOKEN; NO Prolific study, NO humans):
srun -p dev -t 15:00 -c 2 --mem=4G "$VENV_PY" -m src.pipelines.outer_loop.run \
  --project subjective_randomness --experiment 1 \
  --prepare-smoke-experiment --deploy-only \
  --deploy-target firebase --prolific-mode none \
  --firebase-project auto-psych-2c5da --run-label smoke-live
# then: curl -s -o /dev/null -w '%{http_code}\n' https://auto-psych-2c5da.web.app/e1-smoke-live/
```

**Simulated multi-experiment dry-run** (Gemini participants, no humans, no
Firebase) — exercises the `experiment ≥2` posterior-design path before you spend
money on a live sequence. It is a long agentic run, so submit it as a job (copy
`run_live.sbatch` minus the `--mode live --deploy-target firebase --prolific-mode
live` flags, using instead `--experiments 2 --mode
simulated_participants_nobrowser --participant-backend closed`), or ask the
maintainers to run it.

## Output

Per run, under `AUTO_PSYCH_OUTPUT_DIR/<project>/experiment<N>/`:
`cognitive_models/`, `design/stimuli.json`, `experiment/` (jsPsych +
`config.json`), `deployment/deployment_manifest.json`, `data/responses.csv`,
`model_loop/` (posterior + report + proposed models), `model_registry.yaml`.
Slurm logs: `$WORK_ROOT/slurm_logs/`. Browse with
`python -m src.viewer.server --data-root <AUTO_PSYCH_OUTPUT_DIR>`.

## Notes / limits

- **TLS / CA certs**: the uv-managed Python (python-build-standalone) does not
  find el7's system CA store on its own, so without help its `ssl`/`requests`
  fail with `CERTIFICATE_VERIFY_FAILED`. `_env.sh` exports `SSL_CERT_FILE` /
  `REQUESTS_CA_BUNDLE` / `SSL_CERT_DIR` to the system bundle — needed for the
  Prolific API (`requests`) and the `/results` fetch (`urllib`).
- **Browser-based simulation does NOT run on the el7 compute nodes.** Modes that
  drive a headless browser on the cluster — `--mode simulated_participants` *with
  `--deploy-target firebase`* (Firebase browser sim) and the local
  `_collect_from_browser` path — need Playwright, whose bundled Node requires
  glibc ≥ 2.27 (Sherlock has 2.17), so `playwright install` fails. This does
  **not** affect live runs: real Prolific participants use their own browsers;
  the cluster only deploys, manages the study, and fetches results. To generate
  synthetic data on the cluster, use `--mode simulated_participants_nobrowser`
  (Gemini answers stimuli directly — validated here) or the default PyMC
  prior-predictive sampling. For browser sim specifically, run it off-cluster or
  in an Apptainer/Playwright container.
- The live pipeline does **no** server-side Firestore access. Participant data
  flows entirely through the `/submit` and `/results` Cloud Functions
  (`functions/index.js`), which use their own admin credentials to write/read the
  Firestore `responses` subcollection — so the cluster needs only `FIREBASE_TOKEN`
  (deploy) and `PROLIFIC_API_TOKEN` (recruit), not GCP ADC. The deployment record
  is kept in `deployment_manifest.json` on disk. (The separate monitor dashboard
  reads Firestore and needs ADC, but it runs off-cluster.)
- A Firebase Hosting deploy can take a minute or two to propagate across the CDN;
  a freshly deployed `/e{N}-{label}/` page may briefly 404 right after the deploy
  reports success. The live collector polls Prolific for far longer than that, so
  it is not affected.
- `functions/node_modules` is per-worktree and installed on first deploy on the
  compute node (npm cache shared off `$HOME`); `setup.sbatch` warms it.
