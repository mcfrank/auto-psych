# Cloud Run deployment

This document describes how to run the auto-psych pipeline as a **Cloud Run Job** with state in **Firestore** and optional **GCS** for large artifacts. The job syncs project and batch data from Firestore before running, runs `run_pipeline.py` with the same CLI args, then syncs results back. No `.secrets` in the image — use Secret Manager for `GOOGLE_API_KEY`.

## Quick relaunch

**Same project for pipeline and hosting** (e.g. everything in `auto-psych-2c5da`):

```bash
gcloud config set project auto-psych-2c5da
gcloud builds submit --tag gcr.io/auto-psych-2c5da/auto-psych-job
gcloud run jobs update pipeline-job --region us-central1 --set-env-vars=FIREBASE_PROJECT=auto-psych-2c5da
```

**Pipeline in one project, hosting in another** (e.g. pipeline in `auto-psych`, experiments on `auto-psych-2c5da`):

```bash
gcloud config set project auto-psych
gcloud builds submit --tag gcr.io/auto-psych/auto-psych-job
gcloud run jobs update pipeline-job --region us-central1 --set-env-vars=FIREBASE_PROJECT=auto-psych-2c5da
```

## Setup checklist (secrets, IDs, parameters)

You can run the **pipeline** (Cloud Run Job, pipeline Firestore, GCS) in one GCP project and deploy **experiments** (Hosting + Cloud Functions) to a **different** Firebase project. Or use one project for both.

| What | Where / value |
|------|----------------|
| **Pipeline GCP project** | Where the Cloud Run Job and pipeline Firestore/GCS live (e.g. `auto-psych`). Build and push the image to this project’s registry: `gcr.io/auto-psych/auto-psych-job`. |
| **Firebase / Hosting project** | Where the web experiment is deployed (Hosting + Cloud Functions for `/submit`, `/results`). Can be the same as the pipeline project or separate (e.g. `auto-psych-2c5da`). |
| **Firestore for participant results** | In the **Hosting** project (default database). The `functions` use that project’s Firestore. Enable Firestore Native mode there if needed. |
| **Firestore for pipeline state** | In the **pipeline** project. Set job env: `PIPELINE_FIRESTORE_DATABASE=pipeline-state` (or `(default)`). |
| **Secret Manager** | In the **pipeline** project. Create secret `GOOGLE_API_KEY`; the job reads it from there. |
| **`.firebaserc`** (repo root) | `{"projects": {"default": "HOSTING_PROJECT_ID"}}` — the Firebase project you deploy to (e.g. `auto-psych-2c5da`). |
| **`firebase.json`** → `hosting.site` | Your **Hosting** project ID (site ID), e.g. `"site": "auto-psych-2c5da"`. |
| **Cloud Run Job env** | `FIREBASE_PROJECT` = **Hosting** project ID (where the implement step deploys; collect uses that experiment URL). `PIPELINE_FIRESTORE_DATABASE`, `PIPELINE_GCS_BUCKET` refer to the **pipeline** project. |
| **Service account (Cloud Run Job)** | Pipeline project’s identity. Needs **Secret Manager**, **Firestore**, (optional) **GCS** in the **pipeline** project. For deploy: **Firebase Hosting Admin** + **Cloud Functions Admin** in the **Hosting** project (same project if hosting there; if hosting is a different project, grant these roles in the Hosting project — see “Pipeline in one project, Hosting in another”). |

### Service account: where it’s set

You don’t specify a service account in the codebase. By default, a Cloud Run Job runs as the **pipeline project’s** default compute service account (`PIPELINE_PROJECT_NUMBER-compute@developer.gserviceaccount.com`). Grant that account: (1) in the **pipeline** project — Secret Manager Secret Accessor, Cloud Datastore User (Firestore), optional Storage Object Admin; (2) in the **Hosting** project — Firebase Hosting Admin and Cloud Functions Admin (if Hosting is a different project, add the pipeline SA as a member in the Hosting project’s IAM with those roles). See “Pipeline in one project, Hosting in another” below.

After deploy, the experiment URL is `https://HOSTING_PROJECT_ID.web.app` (e.g. `https://auto-psych-2c5da.web.app`). The collect step uses `experiment_url` and `results_api_url` from `config.json`.

### Firebase deploy in the job

The image includes the Firebase CLI, `firebase.json`, and `functions/` so the **implement** step can run `firebase deploy --only hosting,functions`. For that to run inside the job you must:

1. **Set `FIREBASE_PROJECT`** in the Cloud Run Job environment to the **Hosting** project ID (the project where you want the experiment URL). Without it you get "Firebase skipped: no project" and collect runs against a non-existent "local" URL.
2. **Grant the job’s service account** permission to deploy to that Hosting project: **Firebase Hosting Admin** and **Cloud Functions Admin** (or **Firebase Admin**). If the Hosting project is the same as the pipeline project, grant these in that project. If Hosting is a **different** project (e.g. pipeline in `auto-psych`, Hosting in `auto-psych-2c5da`), grant the **pipeline** job’s service account these roles **in the Hosting project** — see next subsection.
3. **“No Hosting site” error:** That message refers to the **project the deploy is targeting**. If you’re already hosting from the Hosting project (e.g. `auto-psych-2c5da`), the site exists — the error usually means the job was targeting the **wrong** project (e.g. `FIREBASE_PROJECT` or `.firebaserc` in the image pointed at `auto-psych`, which has no Hosting). Fix: set `FIREBASE_PROJECT` to the Hosting project ID (`auto-psych-2c5da`) and grant the job’s SA deploy rights there; no need to create a site. Only if the Hosting project has **never** had Hosting set up, create the site once: `firebase hosting:sites:create auto-psych-2c5da --project auto-psych-2c5da`.

If you omit `FIREBASE_PROJECT`, the deploy step skips Firebase and the collect step will try "local" mode (browser to a local server), which does not work in the container.

### Pipeline in one project, Hosting in another

If the **pipeline** runs in GCP project `auto-psych` and you want to deploy experiments to a **separate** Firebase project `auto-psych-2c5da`:

1. **Hosting site:** If you already host from `auto-psych-2c5da`, the site exists — skip this. Only if that project has never had Hosting, create it once:  
   `firebase hosting:sites:create auto-psych-2c5da --project auto-psych-2c5da`

2. **Grant the pipeline job’s service account permission to deploy in the Hosting project.** The job runs as the pipeline project’s default compute SA (e.g. `AUTO_PSYCH_PROJECT_NUMBER-compute@developer.gserviceaccount.com`). In the **Hosting** project you need three things:
   - **Firebase Hosting Admin** and **Cloud Functions Admin** (project-level).
   - **Service Account User** on the Hosting project’s *default App Engine/Cloud Functions* service account (`HOSTING_PROJECT_ID@appspot.gserviceaccount.com`). Without this, you get: *"Missing permissions required for functions deploy. You must have permission iam.serviceAccounts.ActAs on service account ...@appspot.gserviceaccount.com"*.

   In Cloud Console (Hosting project): IAM → add the pipeline SA with “Firebase Hosting Admin” and “Cloud Functions Admin”. Then go to **IAM & Admin → Service Accounts**, open `auto-psych-2c5da@appspot.gserviceaccount.com` (App Engine default), **Permissions** tab → Grant access → add the pipeline SA with role **Service Account User**.

   Or with gcloud (run with a principal that can edit the Hosting project):

   ```bash
   # Pipeline project (where the job runs)
   export PIPELINE_PROJECT=auto-psych
   export HOSTING_PROJECT=auto-psych-2c5da
   PIPELINE_NUMBER=$(gcloud projects describe $PIPELINE_PROJECT --format="value(projectNumber)")
   SA_EMAIL="${PIPELINE_NUMBER}-compute@developer.gserviceaccount.com"

   # Project-level: Hosting + Functions deploy
   gcloud projects add-iam-policy-binding $HOSTING_PROJECT \
     --member="serviceAccount:${SA_EMAIL}" \
     --role="roles/firebasehosting.admin"
   gcloud projects add-iam-policy-binding $HOSTING_PROJECT \
     --member="serviceAccount:${SA_EMAIL}" \
     --role="roles/cloudfunctions.admin"

   # ActAs the default App Engine SA (required for functions deploy)
   gcloud iam service-accounts add-iam-policy-binding ${HOSTING_PROJECT}@appspot.gserviceaccount.com \
     --project=$HOSTING_PROJECT \
     --member="serviceAccount:${SA_EMAIL}" \
     --role="roles/iam.serviceAccountUser"
   ```

3. **Job env:** Set `FIREBASE_PROJECT=auto-psych-2c5da` so the deployer targets the Hosting project. Build and run the job in the pipeline project: `gcr.io/auto-psych/auto-psych-job`, job created in `auto-psych`.

**Alternative:** You can instead enable Firebase in the **pipeline** project (`auto-psych`), create a Hosting site there, and set `FIREBASE_PROJECT=auto-psych`. Then the job deploys to the same project it runs in; no cross-project IAM. Use that if you prefer to keep pipeline and experiments in one project.

## Prerequisites

- A GCP project with **Cloud Run**, **Firestore**, and (optional) **Cloud Storage** enabled. Enable Cloud Storage if you use a GCS bucket for references and large artifacts (recommended).
- Two Firestore databases: **(default)** for participant results (submit/results), and a second (e.g. `pipeline-state`) for pipeline state if you want them separate.
- **Secret Manager**: Create a secret named `GOOGLE_API_KEY` with your Gemini API key (e.g. in [Secret Manager](https://console.cloud.google.com/security/secret-manager)).
- A **service account** for the job with the roles in the table above.

## 1. Build and push the image

From the repo root, use the **pipeline** project (where the job runs). Set gcloud to that project so the build and push both happen there:

```bash
gcloud config set project auto-psych   # or your pipeline project
gcloud builds submit --tag gcr.io/auto-psych/auto-psych-job
```

The build uses `.gcloudignore`, so `projects/`, `.git`, and other unneeded paths are excluded; project state is loaded from Firestore at run time.

### Troubleshooting: "Permission artifactregistry.repositories.uploadArtifacts denied" or push denied

- **Build runs in your default project, push goes to the project in the image tag.** If you run `gcloud builds submit --tag gcr.io/auto-psych-2c5da/auto-psych-job` but your default project is `auto-psych`, the build runs in **auto-psych** and then tries to push to **auto-psych-2c5da**’s registry. The Cloud Build service account in auto-psych cannot push to another project’s registry → permission denied.
- **Fix: use one project.** Set gcloud to the same project as the image tag (and as Firebase):  
  `gcloud config set project auto-psych-2c5da`  
  Then run:  
  `gcloud builds submit --tag gcr.io/auto-psych-2c5da/auto-psych-job`  
  The build and push now both happen in auto-psych-2c5da. Create/update the Cloud Run Job in that project too.
- **If you really use only one project** and still see push denied: the **Cloud Build service account** in that project (e.g. `PROJECT_NUMBER@cloudbuild.gserviceaccount.com`) needs permission to write to the registry. Grant **Storage Admin** (for gcr.io) or **Artifact Registry Writer** (for Artifact Registry) in that same project:

  ```bash
  export PROJECT_ID=auto-psych-2c5da   # or your single project ID
  PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/storage.admin"
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
  --args="--project=subjective_randomness,--runs=12,--mode=simulated_participants,--n-participants=3,--max-retries=5"

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
