# Cloud Run Job (pipeline)

Run the auto-psych pipeline as a Cloud Run Job with state in Firestore. **For full setup, checklist, GCS bucket, Firestore populate, and troubleshooting, see [CLOUDRUN_README.md](../CLOUDRUN_README.md)** in the repo root.

## Prerequisites

- GCP project with Cloud Run, Firestore, and (optional) GCS enabled
- A **new** Firestore database for pipeline state (separate from any Firebase experiment Firestore)
- Secret Manager secret for `GOOGLE_API_KEY`
- Service account with: Firestore read/write, Secret Manager secret accessor, (optional) GCS Storage Object Admin

## Build and push image

```bash
# From repo root
gcloud builds submit --tag gcr.io/PROJECT_ID/auto-psych-job
# Or use Artifact Registry: --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/auto-psych-job
```

`.gcloudignore` excludes `projects/`, `.git`, and other unneeded files from the build context.

## Create or update the job

Inject `GOOGLE_API_KEY` from Secret Manager (do not put the key in env value):

```bash
gcloud run jobs create pipeline-job \
  --image gcr.io/PROJECT_ID/auto-psych-job \
  --region us-central1 \
  --set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest \
  --memory 4Gi --cpu 2 \
  --task-timeout 7200 \
  --max-retries 0
```

Optional env for the job:

- `PIPELINE_FIRESTORE_DATABASE` – Firestore database id (default: `(default)`)
- `PIPELINE_GCS_BUCKET` – GCS bucket for large artifacts (leave unset to store only in Firestore, subject to 1MB doc limit)

## Execute the job

Pass pipeline args as job args (comma-separated in one string or multiple `--args`):

```bash
gcloud run jobs execute pipeline-job --region us-central1 \
  --args="--project=subjective_randomness,--runs=3,--mode=simulated_participants,--n-participants=10,--max-retries=5"
```

For a single run or append:

```bash
gcloud run jobs execute pipeline-job --region us-central1 \
  --args="--project=subjective_randomness,--run=1,--mode=simulated_participants"
gcloud run jobs execute pipeline-job --region us-central1 \
  --args="--project=subjective_randomness,--append,--runs=4-6,--mode=simulated_participants"
```

## Job metadata in Firestore

Each batch document stores full job invocation metadata: `mode`, `n_participants`, `max_retries`, `append`, `runs_spec`, plus `commit_hash` and `dirty` from the codebase at run time.
