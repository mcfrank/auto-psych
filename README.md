# Auto-psych: LangGraph agentic experiment loop

A 6-agent pipeline for psychology experiments: theory → design → implement (includes deploy) → collect data → analyze → interpret. State is file-based under `projects/<phenomenon>/run<N>/<agent>/`.

## Agents

| # | Agent | Role | Key outputs |
|---|-------|------|-------------|
| 1 | **Theory** | Adds **one theory per LLM call**. Run 1: 2–3 calls to add 2–3 models; run 2+: copies previous run’s theories and adds at least one new (1+ calls). Each call outputs one YAML model + one Python file, then ---DONE--- or ---ADD_ANOTHER---. | `1_theory/models_manifest.yaml`, `1_theory/*.py`, `model_registry.yaml` (run dir) |
| 2 | **Design** | Generates a design script that scores stimuli by expected information gain (EIG) using the registry’s theory probabilities; selects top stimuli. | `2_design/stimuli.json`, `design_rationale.md` |
| 3 | **Implement** | Fills the jsPsych template; then runs deploy step (local server or Firebase), writes config. | `3_implement/index.html`, `stimuli.json`, `config.json` |
| 4 | **Collect data** | Simulated mode: opens the experiment in a browser (or Firebase), collects responses and writes CSV. Real mode not yet implemented. | `4_collect/responses.csv` |
| 5 | **Analyze** | Aggregates responses by stimulus, computes summary statistics (Python only). | `5_analyze/aggregate.csv`, `summary_stats.json` |
| 6 | **Interpret** | Compares data (merged over runs 1..n) to model predictions; writes a report and updates theory probabilities in the run's registry. | `6_interpret/report.md`, `theory_probabilities.yaml`, `model_registry.yaml` (updated) |

After each agent runs, a **validation** step runs automatically. If validation fails, the agent is re-invoked with feedback (up to 3 retries), then the pipeline continues to the next agent.

### Observability and debugging

Each agent writes to its run directory:

- **`observability.log`** — Timestamped log of what the agent did: start/end, validation feedback (on retries), LLM invocation, file writes, and any errors. When validation fails, the validator also appends the failure reason here.
- **`transcripts/`** — For agents that call the LLM (theory, design, interpret), one Markdown file per attempt: `transcripts/attempt_001.md`, etc. Each file contains the system prompt, user message, and full LLM response (and any validation feedback for that attempt), so you can see exactly what was sent and why it might have failed or timed out.

The theory agent is called iteratively (one model per call) to reduce validation failures; each call has a 5-minute timeout.

## Requirements

- **Python 3.11+** (recommended)
- **Node.js 14+** (optional; for jsPsych Builder when building/packaging experiments)
- **Firebase** (optional): for collect step with no local server — Hosting + Cloud Functions + Firestore. Otherwise the pipeline uses a local HTTP server.

## Installation

### 1. Clone and enter the project

```bash
cd auto-psych
```

### 2. Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Secrets and config files

Three different things are involved; only one holds API keys.

| File | Purpose | Secret? | Do you need it? |
|------|--------|--------|------------------|
| **`.secrets`** | Holds **API keys** (and optional overrides) the pipeline reads at runtime. | Yes — gitignored. | Yes. Create it and add `GOOGLE_API_KEY` for the LLM agents. |
| **`.firebaserc`** | Tells the **Firebase CLI** which Firebase project to use (`default` project ID). | No — project ID is not secret. | Only if you use Firebase deploy; set `"default": "your-firebase-project-id"`. |
| **`firebase.json`** | **Firebase project config**: hosting folder, rewrites to Cloud Functions. | No. | Yes if you use Firebase; the repo includes a default. Set `hosting.site` to your project ID when deploying manually. |

**What to put in `.secrets`**

- **`GOOGLE_API_KEY`** (required for theory, design, implement, collect steering, interpret): Get an API key for Gemini from [Google AI Studio](https://aistudio.google.com/apikey). The pipeline reads this from the `.secrets` file or from the `GOOGLE_API_KEY` environment variable.
- **`PROLIFIC_API_TOKEN`** (required for `--mode test_prolific` or live Prolific): Create a token in [Prolific Researcher Settings](https://app.prolific.com/researcher/). Used to create test participants and studies via the [Prolific API](https://docs.prolific.com/api-reference/).

Example `.secrets` (do not commit):

```
GOOGLE_API_KEY=your_gemini_api_key_here
PROLIFIC_API_TOKEN=your_prolific_token_here
```

You can use a `.secrets` directory with one file per key (e.g. `.secrets/GOOGLE_API_KEY`) if you prefer.

**Firebase:** The deploy step does **not** read any Firebase API key from `.secrets`. Firebase deploy uses the CLI and your login (`firebase login`). The project is chosen from `.firebaserc` (or the `FIREBASE_PROJECT` environment variable). So you need `.firebaserc` (or `FIREBASE_PROJECT`) and `firebase.json` for Firebase; no Firebase keys in `.secrets`.

### 4. (Optional) Firebase for collect (simulated) with no local server

To run simulated participants against a **deployed experiment** (no local server):

1. **Firebase project**: Create a project at [Firebase Console](https://console.firebase.google.com/), enable Hosting and Firestore.
2. **CLI**: `npm install -g firebase-tools` then `firebase login`. In the repo root, set your project in `.firebaserc`: `"default": "your-project-id"` (or set env `FIREBASE_PROJECT`). If you deploy with `firebase deploy` manually, set the same project ID in `firebase.json` under `hosting.site` (the implement step sets this for you).
3. **Deploy**: Run the pipeline with `--mode simulated_participants`. The implement step copies the experiment to `public/`, deploys with `firebase deploy --only hosting,functions`, and writes `experiment_url` and `results_api_url` in `3_implement/config.json`. The collect step runs simulated (browser or model sampling) or live (Prolific + Firebase when implemented); in simulated mode it opens the experiment URL N times (each run POSTs to `/submit`), then fetches GET `/results` and writes `responses.csv`.

If Firebase is not configured (no real project in `.firebaserc`), the deploy step uses the **local server** at `http://127.0.0.1:8765` and the collect step gathers data from `window.__experimentData` in the browser.

### 5. (Optional) Test Prolific end-to-end

To test the Prolific flow with a single test participant (no real recruitment):

1. **Prolific API token**: Add `PROLIFIC_API_TOKEN` to `.secrets` (see above). The [test participant API](https://docs.prolific.com/api-reference/users/create-test-participant-for-researcher) must be enabled for your Prolific workspace.
2. **Project config**: Create `projects/<project_id>/prolific_config.yaml` (or copy from `prolific_config.yaml.example`). Set **`test_participant_email`** to an email that is **not** already registered on Prolific; this account will be created as a test participant and used as the only place in the study.
3. **Run**: `python3 run_pipeline.py --project <project_id> --run 1 --mode test_prolific`. The implement step deploys to Firebase, creates the test participant, creates a 1-place study with that participant on a custom allowlist, and publishes it. The collect step polls Prolific until the study is complete, then fetches results from Firebase. After the participant finishes the experiment, the page redirects to Prolific’s completion URL.

### 6. (Optional) Cloud Run Job deployment

You can run the full pipeline as a **Cloud Run Job** with state stored in a **Firestore** database (and optional GCS for large artifacts). The job syncs project and batch data from Firestore before running, runs `run_pipeline.py` with the same CLI args, then syncs results back. No `.secrets` in the image — use Secret Manager for `GOOGLE_API_KEY`.

**Setup checklist (secrets, IDs, parameters)**

Use one GCP project for both **participant results** (Firebase Hosting + Cloud Functions + Firestore) and **pipeline state** (Cloud Run Job + Firestore). With two Firestore databases, use one for pipeline state and the **(default)** database for participant results (the existing Cloud Functions use the default DB).

| What | Where / value |
|------|----------------|
| **GCP Project ID** | Your project ID (e.g. `my-auto-psych`). Use this everywhere below. |
| **Firestore for participant results** | Use the **(default)** database. The `functions` (submit/results) use `admin.firestore()` with no database id, so they always write/read the default DB. Enable Firestore in Native mode if you haven’t. |
| **Firestore for pipeline state** | Your second database (e.g. `pipeline-state`). Set as job env: `PIPELINE_FIRESTORE_DATABASE=pipeline-state` (omit or use `(default)` if you use the default DB for pipeline instead). |
| **Secret Manager** | Create secret **name** `GOOGLE_API_KEY`, value = your Gemini API key. The Cloud Run Job references it as `GOOGLE_API_KEY:latest`. |
| **`.firebaserc`** (repo root) | `{"projects": {"default": "YOUR_GCP_PROJECT_ID"}}` so the deploy step uses this project for Hosting + Functions. |
| **`firebase.json`** → `hosting.site` | Set to your **Firebase Hosting site** (usually the same as GCP Project ID). Example: `"site": "my-auto-psych"`. |
| **Cloud Run Job env (optional)** | `PIPELINE_FIRESTORE_DATABASE` = your pipeline Firestore database id (e.g. `pipeline-state`). `PIPELINE_GCS_BUCKET` = bucket name if you use GCS for large artifacts (optional). |
| **Service account (Cloud Run Job)** | The job runs as an identity that needs **Secret Manager Secret Accessor**, **Cloud Datastore User** (Firestore), and (if using GCS) **Storage Object Admin**. You don’t put this in the repo — see below. |

**Service account: where it’s set**

You don’t specify a service account in the codebase. By default, a Cloud Run Job runs as the project’s **default compute service account** (`PROJECT_NUMBER-compute@developer.gserviceaccount.com`). When you run `gcloud run jobs create ...` without `--service-account`, Cloud Run uses that default. You only need to **grant that account** the roles above (Console: **IAM & Admin → IAM**, find “Compute Engine default service account”, add the roles; or use `gcloud projects add-iam-policy-binding` with `--member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com"`). If you prefer a dedicated service account, create one in **IAM & Admin → Service Accounts**, grant it the same roles, then pass it when creating the job: `--service-account=your-sa@PROJECT_ID.iam.gserviceaccount.com`.

After deploy, the experiment URL is `https://YOUR_GCP_PROJECT_ID.web.app` (or your custom domain). The collect step uses `experiment_url` and `results_api_url` from `config.json` (same origin for `/submit` and `/results`).

**Prerequisites**

- A GCP project with **Cloud Run**, **Firestore**, and (optional) **Cloud Storage** enabled. Enable Cloud Storage if you use a GCS bucket for references and large artifacts (recommended).
- Two Firestore databases: **(default)** for participant results (submit/results), and a second (e.g. `pipeline-state`) for pipeline state if you want them separate.
- **Secret Manager**: Create a secret named `GOOGLE_API_KEY` with your Gemini API key (e.g. in [Secret Manager](https://console.cloud.google.com/security/secret-manager)).
- A **service account** for the job with the roles in the table above.

**1. Build and push the image**

From the repo root:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/auto-psych-job
```

Replace `PROJECT_ID` with your GCP project ID. The build uses `.gcloudignore`, so `projects/`, `.git`, and other unneeded paths are excluded; project state is loaded from Firestore at run time.

**Troubleshooting: "Permission artifactregistry.repositories.uploadArtifacts denied"**

- **Who pushes the image:** Your `gcloud` login is only used to *submit* the build to Cloud Build. The actual *push* of the image is done by the **Cloud Build service account** in your project (e.g. `PROJECT_NUMBER@cloudbuild.gserviceaccount.com`), not your user account. The denial means that service account doesn’t have permission to upload to the image registry.
- **Check your gcloud:** Run `gcloud auth list` to see which account is active and `gcloud config get-value project` to see the current project. Log in with `gcloud auth login` and set the project with `gcloud config set project PROJECT_ID`.
- **Fix the push permission:** Grant the Cloud Build service account permission to write to the registry. For `gcr.io` (and when Cloud Build uses Artifact Registry under the hood), grant **Artifact Registry Writer** to the Cloud Build default service account:

  ```bash
  # Get your project number (not the project ID)
  gcloud projects describe PROJECT_ID --format="value(projectNumber)"
  # Grant Artifact Registry Writer to the Cloud Build service account
  gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
  ```

  Replace `PROJECT_ID` and `PROJECT_NUMBER`. If the image lives in a specific Artifact Registry repo (e.g. you switched to `REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/...`), you can limit the role to that repository in the IAM page instead of project-wide.

**2. Create the Cloud Run Job**

Create the job with `GOOGLE_API_KEY` injected from Secret Manager (do not put the key in an env value):

```bash
gcloud run jobs create pipeline-job \
  --image gcr.io/auto-psych/auto-psych-job \
  --region us-central1 \
  --set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest \
  --memory 4Gi --cpu 2 \
  --task-timeout 7200 \
  --max-retries 0
```

Adjust `--region` if needed. If you use a second Firestore database for pipeline state (see checklist above), add e.g. `--set-env-vars=PIPELINE_FIRESTORE_DATABASE=pipeline-state`. Other optional env:

- `PIPELINE_FIRESTORE_DATABASE` — Firestore database id for pipeline state (default: `(default)`).
- `PIPELINE_GCS_BUCKET` — GCS bucket name for large artifacts (references, run outputs); recommended so large reference files and run artifacts are stored in GCS instead of hitting Firestore’s 1MB doc limit.

**2b. Create a GCS bucket for references and large artifacts (recommended)**

Create a bucket and grant the Cloud Run job’s service account access so large reference files and run artifacts can be stored in GCS. Replace `auto-psych` with your actual GCP project ID.

```bash
# Set your project ID (required — do not leave as PROJECT_ID)
export PROJECT_ID=auto-psych
export PIPELINE_GCS_BUCKET=auto-psych-results-docs
gsutil mb -p $PROJECT_ID -l us-central1 gs://${PIPELINE_GCS_BUCKET}/

# Grant the default compute SA (used by the Cloud Run Job) write/read to the bucket
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gsutil iam ch serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com:objectAdmin gs://${PIPELINE_GCS_BUCKET}
```

When you run the populate script (step 3), set the same bucket so large references are uploaded to GCS: `export PIPELINE_GCS_BUCKET=auto-psych-pipeline-YOUR_PROJECT_ID` (or your bucket name). When creating or updating the job, add `--set-env-vars=PIPELINE_GCS_BUCKET=your-bucket-name`.

**3. Populate Firestore with a project (one-time)**

Before the first run, the project must exist in Firestore so the job can sync it down. From your machine you need **Application Default Credentials** so the script can write to Firestore: run `gcloud auth application-default login` (and choose the account that has access to the project). If you created a GCS bucket (step 2b), set `export PIPELINE_GCS_BUCKET=your-bucket-name` so large reference files are uploaded to GCS. Then run:

```bash
source venv/bin/activate
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.firestore_sync import ensure_project_in_firestore
ensure_project_in_firestore('subjective_randomness', Path('projects/subjective_randomness'))
"
```

Use your project id and path. This writes `problem_definition.md` and `references/` from the local `projects/` tree into Firestore (and uploads large reference files to GCS when `PIPELINE_GCS_BUCKET` is set).

**4. Execute the job**

Pass pipeline arguments as job args (comma-separated). Examples:

```bash
# Multiple runs (creates a batch), 10 simulated participants per run
gcloud run jobs execute pipeline-job --region us-central1 \
  --args="--project=subjective_randomness,--runs=3,--mode=simulated_participants,--n-participants=10,--max-retries=5"

# Single run
gcloud run jobs execute pipeline-job --region us-central1 \
  --args="--project=subjective_randomness,--run=1,--mode=simulated_participants"

# Append runs to the latest batch
gcloud run jobs execute pipeline-job --region us-central1 \
  --args="--project=subjective_randomness,--append,--runs=4-6,--mode=simulated_participants"
```

The job runs to completion (theory → design → implement → collect → analyze → interpret for each run). Batch and run outputs are synced back to Firestore (and optionally GCS). Each batch document stores full job metadata: `mode`, `n_participants`, `max_retries`, `append`, `runs_spec`, and codebase `commit_hash` / `dirty`.

For more detail (YAML spec, updating the job), see [cloudrun/README.md](cloudrun/README.md).

## Running the pipeline

```bash
source venv/bin/activate
python3 run_pipeline.py --project subjective_randomness --run 1 --mode simulated_participants
python3 run_pipeline.py --project subjective_randomness --runs 3 --mode simulated_participants
python3 run_pipeline.py --project subjective_randomness --runs 4-6 --mode simulated_participants
python3 run_pipeline.py --project subjective_randomness --runs 2 --n-participants 10 --max-retries 5
```

- `--project`: Project id (e.g. `subjective_randomness`); must have a `problem_definition.md` under `projects/<project>/`.
- `--run`: Single run number (creates `projects/<project>/run<N>/`). Use this or `--runs`.
- `--runs`: Runs to execute: **N** (runs 1 through N) or **A-B** (runs A through B inclusive, e.g. `4-6`). Each run gets its own directory; the theory agent in run n reads run n−1’s interpreter report and registry; the interpreter in run n sees merged data from runs 1..n. When you use `--runs`, runs are stored in a **batch** directory (see Batch archiving below).
- `--append`: Add runs to the **latest** batch instead of creating a new one. Use with `--run N` or `--runs A-B`. Run IDs are those you specify (e.g. if the batch already has run1–3, use `--append --runs 4-5` to add run4 and run5). Cannot be used with `--agent`.
- `--mode`: `simulated_participants`, `live`, or `test_prolific`. Use `test_prolific` to run the full Prolific flow with one test participant (Firebase deploy + Prolific test participant + study + collect).
- `--n-participants`: Number of simulated participants per run (default: 5). Default is in `src/config.py` as `DEFAULT_SIMULATED_N_PARTICIPANTS`.
- `--max-retries`: Max validation retries per agent before moving on (default: 3). Default in `src/config.py` as `DEFAULT_MAX_VALIDATION_RETRIES`.

Validation runs **inside** the pipeline after each agent: if an agent’s output fails validation, the agent is re-run with the failure message (up to `--max-retries`), then execution continues.

Prompts are resolved from `prompts/` with optional overrides in `projects/<project>/prompts/`. The resolved prompts used for each run are copied to `projects/<project>/run<N>/prompts_used/` (or, when using a batch, to the batch run directory) for reproducibility.

### Batch archiving

When you use **`--runs`** (e.g. `--runs 3` or `--runs 4-6`), the pipeline creates a **batch** directory so that all runs are stored under a timestamp and codebase hash:

- **Path:** `projects/<project_id>/batches/batch_<YYYYMMDD>-<HHMM>_<short_hash>/`
  - Example: `batch_20260316-0923_a1b2c3d`. The short hash is from `git rev-parse HEAD` (or `nogit` / `nogit_dirty` if not a repo).
- **Contents:** `commit_hash.txt` (full commit, timestamp, dirty flag), `run1/`, `run2/`, … (same layout as `projects/<project>/run<N>/`), plus `correlations.csv` (model–data correlation per run) and `correlations_by_run.png` (line plot: x = run, y = correlation, one line per theory).

Use **`--append`** with `--run` or `--runs` to add more runs to the most recent batch instead of creating a new one. The correlation CSV and plot are updated after each run. The plot is the main convergence diagnostic (no separate metric).

## Running a single agent (debugging)

To run one agent in isolation (no validation loop, no next agent), use **`run_pipeline.py --agent`** or the convenience script **`run_agent.py`** (same options):

```bash
python3 run_pipeline.py --project subjective_randomness --run 1 --agent 1_theory
python3 run_agent.py --project subjective_randomness --run 2 --agent 2_design --state-from-run 1
python3 run_agent.py --project subjective_randomness --run 1 --agent 1_theory --use-fixtures
```

- `--agent`: Agent to run (e.g. `1_theory`, `2_design`, `3_implement`, `4_collect`, `5_analyze`, `6_interpret`).
- `--state-from-run R`: Load artifact paths from run `R` so you can re-run or run a downstream agent using outputs from a previous run.
- `--use-fixtures`: Use minimal state with `tests/fixtures/` for missing inputs (no prior full run required).
- `--n-participants`, `--max-retries`: Same as pipeline (used when running implement or for consistency).

Use `python3` in the examples; on many systems `python` points to Python 2 or is missing. After `source venv/bin/activate`, `python` usually works too (venv’s `python` is Python 3).

`run_agent.py` is a thin wrapper that calls `run_pipeline.py`; all parameters are defined in `run_pipeline.py`.

## Validating outputs (standalone critic)

Validation is **built into the pipeline**: after each agent, the same checks run automatically and failed agents are re-invoked with feedback (see Agents above). You can also run the validators **standalone** (e.g. to audit a past run without re-running agents):

```bash
python3 run_critic.py --project subjective_randomness --run 1
python3 run_critic.py --project subjective_randomness --run 1 --agent 1_theory
```

- Exits 0 if all validations pass, 1 if any fail.
- Use `--no-write` to only print results and not write `validation.json` under each agent dir.
- The same validator logic is used by the pipeline’s built-in validation loop.

## Tests

Tests are run **manually**. Install dev dependencies and run pytest:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests include: model library (each model returns a probability distribution), EIG bounds and analytic cases, correlation helpers (Pearson r, model–data correlations), theory/design output validators, state loader, and a single-agent run (theory with fixtures) followed by validation. You can add CI (e.g. GitHub Actions) to run `pytest` on push or PR.

## Project layout

```
prompts/                 # Canonical agent prompts (one file per agent)
projects/
  <project_id>/
    problem_definition.md   # Human-authored: task + stimulus schema
    prolific_config.yaml    # Optional; for test_prolific/live (completion_code, test_participant_email, etc.)
    references/             # Optional PDFs referenced by problem_definition.md
    prompts/                # Optional project-specific prompt overrides
    run<N>/
      prompts_used/        # Frozen prompts used in this run
      1_theory/
      2_design/
      3_implement/
      4_collect/
      5_analyze/
      6_interpret/
.secrets                  # Gitignored; see above
```

## Observability

- **Prompts used per run**: For each run, the resolved prompts (canonical + any project overrides) are copied to `projects/<project_id>/run<N>/prompts_used/` (one `.md` per agent and a `manifest.json`). This gives a reproducible record of which prompt text each agent used.
- **Artifacts**: Each agent writes its outputs under `projects/<project_id>/run<N>/<agent>/` (e.g. `1_theory/models_manifest.yaml`, `5_analyze/aggregate.csv`). Inspect these to debug or reproduce a run.
- **Design agent**: Writes `2_design/designer_observability.log` with timestamped steps (LLM call, fenced-block extraction, script run, fallback reason). If the design step falls back, see that log and `design_script_log.txt` (script stdout/stderr or exception) for the cause.

## Code style and docs

- Python: Black/Ruff recommended; docstrings for public APIs.
- README and in-code comments document setup and usage.
