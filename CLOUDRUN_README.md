# Cloud Run deployment

This document describes how to run the auto-psych pipeline as a **Cloud Run Job** with state in **Firestore** and optional **GCS** for large artifacts. The job syncs project and batch data from Firestore before running, runs `run_pipeline.py` with the same CLI args, then syncs results back. No `.secrets` in the image — use Secret Manager for `GOOGLE_API_KEY`.

## Setup checklist (secrets, IDs, parameters)

Use one GCP project for both **participant results** (Firebase Hosting + Cloud Functions + Firestore) and **pipeline state** (Cloud Run Job + Firestore). With two Firestore databases, use one for pipeline state and the **(default)** database for participant results (the existing Cloud Functions use the default DB).

| What | Where / value |
|------|----------------|
| **GCP Project ID** | Your project ID (e.g. `my-auto-psych`). Use this everywhere below. |
| **Firestore for participant results** | Use the **(default)** database. The `functions` (submit/results) use `admin.firestore()` with no database id, so they always write/read the default DB. Enable Firestore in Native mode if you haven’t. |
| **Firestore for pipeline state** | Your second database (e.g. `pipeline-state`). Set as job env: `PIPELINE_FIRESTORE_DATABASE=pipeline-state` (omit or use `(default)` if you use the default DB for pipeline instead). |
| **Secret Manager** | Create secret **name** `GOOGLE_API_KEY`, value = your Gemini API key. The Cloud Run Job references it as `GOOGLE_API_KEY:latest`. |
| **`.firebaserc`** (repo root) | `{"projects": {"default": "YOUR_GCP_PROJECT_ID"}}` so the deploy step uses this project for Hosting + Functions. |
| **`firebase.json`** → `hosting.site` | Set to your **Firebase Hosting site** (usually the same as GCP Project ID). Example: `"site": "my-auto-psych"`. |
| **Cloud Run Job env** | `FIREBASE_PROJECT` = your GCP project ID (required for deploy/collect: implement step deploys to Firebase, collect uses the deployed URL). `PIPELINE_FIRESTORE_DATABASE`, `PIPELINE_GCS_BUCKET` as needed. |
| **Service account (Cloud Run Job)** | The job runs as an identity that needs **Secret Manager Secret Accessor**, **Cloud Datastore User** (Firestore), (if using GCS) **Storage Object Admin**, and (for Firebase deploy) **Firebase Hosting Admin** + **Cloud Functions Admin** (or **Firebase Admin**). See below. |

### Service account: where it’s set

You don’t specify a service account in the codebase. By default, a Cloud Run Job runs as the project’s **default compute service account** (`PROJECT_NUMBER-compute@developer.gserviceaccount.com`). When you run `gcloud run jobs create ...` without `--service-account`, Cloud Run uses that default. You only need to **grant that account** the roles above (Console: **IAM & Admin → IAM**, find “Compute Engine default service account”, add the roles; or use `gcloud projects add-iam-policy-binding` with `--member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com"`). If you prefer a dedicated service account, create one in **IAM & Admin → Service Accounts**, grant it the same roles, then pass it when creating the job: `--service-account=your-sa@PROJECT_ID.iam.gserviceaccount.com`.

After deploy, the experiment URL is `https://YOUR_GCP_PROJECT_ID.web.app` (or your custom domain). The collect step uses `experiment_url` and `results_api_url` from `config.json` (same origin for `/submit` and `/results`).

### Firebase deploy in the job

The image includes the Firebase CLI, `firebase.json`, and `functions/` so the **implement** step can run `firebase deploy --only hosting,functions`. For that to run inside the job you must:

1. **Set `FIREBASE_PROJECT`** in the Cloud Run Job environment to your GCP project ID (the deployer reads this when `.firebaserc` is not present in the container). Without it you get "Firebase skipped: no project" and collect runs against a non-existent "local" URL.
2. **Grant the job’s service account** permission to deploy: e.g. **Firebase Hosting Admin** and **Cloud Functions Admin** (or the broader **Firebase Admin**). The CLI uses Application Default Credentials (the job’s service account) when running in Cloud Run.

If you omit `FIREBASE_PROJECT`, the deploy step skips Firebase and the collect step will try "local" mode (browser to a local server), which does not work in the container.

## Prerequisites

- A GCP project with **Cloud Run**, **Firestore**, and (optional) **Cloud Storage** enabled. Enable Cloud Storage if you use a GCS bucket for references and large artifacts (recommended).
- Two Firestore databases: **(default)** for participant results (submit/results), and a second (e.g. `pipeline-state`) for pipeline state if you want them separate.
- **Secret Manager**: Create a secret named `GOOGLE_API_KEY` with your Gemini API key (e.g. in [Secret Manager](https://console.cloud.google.com/security/secret-manager)).
- A **service account** for the job with the roles in the table above.

## 1. Build and push the image

From the repo root:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/auto-psych-job
```

Replace `PROJECT_ID` with your GCP project ID. The build uses `.gcloudignore`, so `projects/`, `.git`, and other unneeded paths are excluded; project state is loaded from Firestore at run time.

### Troubleshooting: "Permission artifactregistry.repositories.uploadArtifacts denied"

- **Who pushes the image:** Your `gcloud` login is only used to *submit* the build to Cloud Build. The actual *push* of the image is done by the **Cloud Build service account** in your project (e.g. `PROJECT_NUMBER@cloudbuild.gserviceaccount.com`), not your user account. The denial means that service account doesn’t have permission to upload to the image registry.
- **Check your gcloud:** Run `gcloud auth list` to see which account is active and `gcloud config get-value project` to see the current project. Log in with `gcloud auth login` and set the project with `gcloud config set project PROJECT_ID`.
- **Fix the push permission:** Grant the Cloud Build service account permission to write to the registry. Replace `PROJECT_ID` and `PROJECT_NUMBER` with your values:

  ```bash
  export PROJECT_ID=auto-psych
  PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
  ```

### Troubleshooting: "PERMISSION_DENIED" when running gcloud builds submit

Your user account needs permission to trigger Cloud Build. Grant yourself **Cloud Build Editor** (or have a project owner do it):

```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="user:YOUR_EMAIL@gmail.com" \
  --role="roles/cloudbuild.builds.editor"
```

## 2. Create the Cloud Run Job

Create the job with `GOOGLE_API_KEY` injected from Secret Manager (do not put the key in an env value). Set `FIREBASE_PROJECT` so the implement step deploys to Firebase and the collect step can use the deployed experiment URL:

```bash
gcloud run jobs create pipeline-job \
  --image gcr.io/PROJECT_ID/auto-psych-job \
  --region us-central1 \
  --set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest \
  --set-env-vars=FIREBASE_PROJECT=PROJECT_ID \
  --memory 4Gi --cpu 2 \
  --task-timeout 7200 \
  --max-retries 0
```

Replace `PROJECT_ID` with your GCP project ID in both the image path and `FIREBASE_PROJECT`. Add more env as needed: `PIPELINE_FIRESTORE_DATABASE=pipeline-state`, `PIPELINE_GCS_BUCKET=your-bucket-name`.

**If the job already exists** and you only need to add `FIREBASE_PROJECT`, update the job:  
`gcloud run jobs update pipeline-job --region us-central1 --set-env-vars=FIREBASE_PROJECT=your-project-id`

Env vars:

- `FIREBASE_PROJECT` — **Required for deploy/collect.** Your GCP project ID so the implement step deploys to Firebase Hosting/Functions and collect uses the deployed experiment URL.
- `PIPELINE_FIRESTORE_DATABASE` — Firestore database id for pipeline state (default: `(default)`).
- `PIPELINE_GCS_BUCKET` — GCS bucket name for large artifacts (references, run outputs); recommended so large reference files and run artifacts are stored in GCS instead of hitting Firestore’s 1MB doc limit.

## 2b. Create a GCS bucket for references and large artifacts (recommended)

Create a bucket and grant the Cloud Run job’s service account access. Replace `auto-psych` with your actual GCP project ID.

```bash
# Set your project ID (required — do not leave as PROJECT_ID)
export PROJECT_ID=auto-psych
export PIPELINE_GCS_BUCKET=auto-psych-results-docs
gsutil mb -p $PROJECT_ID -l us-central1 gs://${PIPELINE_GCS_BUCKET}/

# Grant the default compute SA (used by the Cloud Run Job) write/read to the bucket
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gsutil iam ch serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com:objectAdmin gs://${PIPELINE_GCS_BUCKET}
```

When you run the populate script (step 3), set the same bucket: `export PIPELINE_GCS_BUCKET=your-bucket-name`. When creating or updating the job, add `--set-env-vars=PIPELINE_GCS_BUCKET=your-bucket-name`.

## 3. Populate Firestore with a project (one-time)

Before the first run, the project must exist in Firestore so the job can sync it down.

1. **Application Default Credentials:** Run `gcloud auth application-default login` (choose the account that has access to the project).
2. **GCS bucket (if used):** Set `export PIPELINE_GCS_BUCKET=your-bucket-name` so large reference files are uploaded to GCS.
3. **Run the populate script:**

```bash
source venv/bin/activate
pip install -r requirements.txt   # ensures google-cloud-firestore (and gcs) are installed
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.firestore_sync import ensure_project_in_firestore
ensure_project_in_firestore('subjective_randomness', Path('projects/subjective_randomness'))
"
```

Use your project id and path. This writes `problem_definition.md` and `references/` from the local `projects/` tree into Firestore (and uploads large reference files to GCS when `PIPELINE_GCS_BUCKET` is set).

## 4. Execute the job

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

The job runs to completion (theory → design → implement → collect → analyze → interpret for each run). Batch and run outputs are synced back to Firestore (and optionally GCS). Each batch document stores full job metadata: `mode`, `n_participants`, `max_retries`, `append`, `runs_spec`, and codebase `commit_hash` / `dirty` (or `nogit` when git isn’t in the container).

## Where run outputs live in Firestore

Run outputs are **not** under the `projects` collection. Use the **pipeline state** Firestore database (the one you set with `PIPELINE_FIRESTORE_DATABASE`, or `(default)` if that’s your pipeline DB). In that database:

| Collection / path | Contents |
|-------------------|----------|
| **`projects`** | Project-level data only: `problem_definition` and a **references** subcollection (one doc per reference file). Populated by the one-time `ensure_project_in_firestore` script. No run outputs here. |
| **`batches`** | One document per batch (e.g. `batch_20260316-1234_nogit`). Fields: `project_id`, `created_at`, `commit_hash`, `dirty`, `run_ids`, `mode`, `n_participants`, `max_retries`, `append`, `runs_spec`, and optionally `correlations_csv` or `correlations_csv_gcs_uri`, `correlations_plot_gcs_uri`. |
| **`batches/{batch_id}/runs/{run_id}`** | One document per run: `project_id`, `batch_id`, `run_id`, `status`. |
| **`batches/{batch_id}/runs/{run_id}/artifacts/{path}`** | One document per file in that run (e.g. `1_theory/rationale.md`, `4_collect/responses.csv`, `6_interpret/report.md`). Each doc has either `content` (text) or `gcs_uri` (if the file was large and GCS is configured). |

Sync-up runs only **after** the pipeline finishes successfully and only when you run the **full pipeline with a batch** (e.g. `--runs 2`). If the job exits with an error before the end, or you use a single `--run 1` without `--runs`, no batch is created and nothing is written to `batches`. So to see run outputs: run with `--runs N`, ensure the job completes, then look under **batches** in the pipeline-state database.

## YAML spec and updating the job

For a job YAML template and update workflow, see [cloudrun/README.md](cloudrun/README.md).
