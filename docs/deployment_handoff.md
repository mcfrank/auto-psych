# Deployment Handoff

This file is the source of truth for deployment tasks that code cannot complete without external credentials, approvals, or project-specific decisions.

## Status At A Glance

### Implemented In Code

- Deployment package at `src/pipelines/outer_loop/deployment/`.
- Dry-run deployment that stages an experiment, Firebase config, client config, and deployment manifest without external network calls.
- Firebase deployment path for Hosting + Cloud Functions.
- Firebase Functions `/submit` endpoint that writes participant responses to Firestore.
- Firebase Functions `/results` endpoint that exports response CSV by `collection_session_id`, with legacy `project_id`/`run_id` fallback.
- Session-centered Firestore response path:

```text
collection_sessions/{collection_session_id}/responses/{participant_id}
```

- Deployment manifest and public `auto_psych_config.json` containing deployment/session/agent bookkeeping.
- Outer-loop CLI flags for deployment:

```text
--deploy-target {none,dry-run,firebase}
--collection-owner <id>
--firebase-project <id>
--firebase-region <region>
--prolific-mode {none,test,live}
--deploy-only
--prepare-smoke-experiment
```

- Prolific payload construction and low-level Prolific API wrappers.
- Tests for manifest creation, Firebase staging, Firestore payload/CSV shaping, Prolific payload construction, and dry-run smoke deployment.

### What Works Now

- Dry-run deployment works locally without Firebase or Prolific credentials.
- Firebase Hosting + Functions deploy worked for the smoke experiment.
- A browser participant submission wrote trial data to Firestore through `/submit`.
- `/results?collection_session_id=...` returned CSV from Firestore.

### Not Yet Fully Proven

- Prolific test-mode launch with a real Prolific API token/workspace.
- Production Prolific launch with real participant budget.
- End-to-end outer-loop run from `3_implement` to Firebase deploy to `4_collect` on a non-smoke generated experiment.
- Python-side Firestore deployment/session metadata writes using local Application Default Credentials on every collaborator laptop.
- Production IAM/billing/region expectations beyond the tested smoke deploy.

### External TODOs Remain

- Confirm Firebase billing/IAM/region expectations with Mike.
- Finish Stanford-account Application Default Credentials or use a service account for Python metadata writes.
- Get Prolific token/workspace access.
- Confirm Prolific reward, participant count, title/description, completion code, IRB/consent, and launch approval.

## External TODOs

- [x] Get Mike's Firebase project id: `auto-psych-2c5da`.
- [x] Create local `.firebaserc` pointing at `auto-psych-2c5da`.
- [x] Log in to Google CLI with Stanford credentials.
- [ ] Set up Google Application Default Credentials for Python Firestore metadata writes, with the Stanford account as the active/default account.
- [x] Confirm Firebase Hosting, Cloud Functions, and Firestore work for the smoke test.
- [ ] Confirm billing status and production IAM expectations with Mike.
- [ ] Confirm the deploy account has IAM permissions for Hosting, Functions, Firestore, and service-account ActAs where required.
- [ ] Confirm the Firebase Functions region. Default in this repo is `us-central1`.
- [ ] Get Prolific API token and workspace access.
- [ ] Confirm Prolific participant count, reward, study name, study description, and estimated completion time.
- [ ] Confirm IRB/consent text, completion-code language, and redirect behavior.
- [ ] Get explicit approval before launching a production Prolific study.

## First-Time Tool Install

On a new macOS laptop, install the Google Cloud CLI, Node/npm, and Firebase CLI
before running the verification commands below.

```bash
brew update
brew install --cask google-cloud-sdk
brew install node
npm install -g firebase-tools
```

Open a new terminal after installing `google-cloud-sdk`. If `gcloud` is still not
found, source the SDK shell helpers from the install path shown by Homebrew, then
restart the terminal.

Verify the commands exist:

```bash
gcloud --version
node --version
npm --version
firebase --version
```

Then authenticate:

```bash
gcloud auth login
gcloud config set project auto-psych-2c5da
firebase login
```

## Local Files To Create

The local `.firebaserc` has been created in this workspace:

```json
{
  "projects": {
    "default": "auto-psych-2c5da"
  }
}
```

It is intentionally ignored by git. If setting up another machine, copy the
example Firebase config and replace the placeholder with Mike's project:

```bash
cp .firebaserc.example .firebaserc
```

The committed `firebase.json` is configured for Hosting site
`auto-psych-2c5da`, public deploy directory `public`, and Cloud Functions
rewrites for `/submit` and `/results` in `us-central1`. Functions deploy on
Node.js 22.

Real secrets should stay out of git. Use environment variables, a local ignored
`.secrets`, or Google Secret Manager.

Expected local Prolific token:

```bash
export PROLIFIC_API_TOKEN=...
```

Optional collection owner identity:

```bash
export AUTO_PSYCH_COLLECTION_OWNER=linas
```

## Step-By-Step Deployment Test

### 1. Verify Firebase/GCP Access

Run:

```bash
gcloud auth list
gcloud config set account linasmn@stanford.edu
gcloud config get-value account
gcloud config get-value project
gcloud auth application-default login --account=linasmn@stanford.edu
gcloud auth application-default set-quota-project auto-psych-2c5da
firebase login:list
firebase projects:list
firebase use
```

Expected:

- `gcloud config get-value account` is `linasmn@stanford.edu`.
- `gcloud config get-value project` is `auto-psych-2c5da`.
- Application Default Credentials are set with the Stanford account for Python Firestore metadata writes.
- `firebase use` shows `default (auto-psych-2c5da)`.
- `firebase projects:list` includes `auto-psych-2c5da`.

### 2. Dry-Run Deploy With A Smoke Experiment

This avoids the `claude`/`opencode` dependency and tests only the deployment
machinery. It writes a tiny jsPsych experiment, stages Firebase deploy files,
and writes deployment bookkeeping locally. It does not contact Firebase or
Prolific.

```bash
uv run python -m src.pipelines.outer_loop.run \
  --project subjective_randomness \
  --experiment 1 \
  --resume \
  --prepare-smoke-experiment \
  --deploy-only \
  --deploy-target dry-run \
  --collection-owner linas
```

Expected outputs:

- `data/outer_loop/subjective_randomness/experiment1/experiment/index.html`
- `data/outer_loop/subjective_randomness/experiment1/experiment/config.json`
- `data/outer_loop/subjective_randomness/experiment1/deployment/deployment_manifest.json`
- `data/outer_loop/subjective_randomness/experiment1/deployment/public/auto_psych_config.json`

### 3. Real Firebase Smoke Deploy

After dry-run succeeds, deploy the same smoke experiment to Firebase:

```bash
uv run python -m src.pipelines.outer_loop.run \
  --project subjective_randomness \
  --experiment 1 \
  --resume \
  --deploy-only \
  --deploy-target firebase \
  --firebase-project auto-psych-2c5da \
  --firebase-region us-central1 \
  --collection-owner linas
```

Expected:

- Firebase Hosting and Functions deploy successfully.
- The hosted experiment is available at `https://auto-psych-2c5da.web.app`.
- Firestore metadata documents are written for the deployment/session.

The deployer checks `functions/node_modules` and runs
`npm --prefix functions install` automatically if `firebase-functions` or
`firebase-admin` is missing. If Firebase reports
`Couldn't find firebase-functions package in your source code`, rerun the command
after confirming `npm --prefix functions install` succeeds locally.

If the deploy succeeds but metadata writing fails with
`DefaultCredentialsError: Your default credentials were not found`, run:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project auto-psych-2c5da
```

Then rerun the Firebase smoke deploy command.

Application Default Credentials can fail if the browser defaults to a personal
Google account or if the Stanford account does not have the right permission.
Make sure `linasmn@stanford.edu` is the active/default account for both gcloud
and ADC. If Stanford OAuth still blocks Application Default Credentials, the
Firebase deploy can still proceed. The deployment code treats Firestore metadata
writes as non-blocking: Hosting/Functions deploy can succeed, the error is saved
in `deployment_manifest.json`, and participant responses can still be written by
the deployed `/submit` function. For complete deployment metadata, use one of:

- Ask Mike for IAM access that allows ADC for `linasmn@stanford.edu`.
- Ask Mike for a service-account JSON key with Firestore write permissions and run:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Then rerun the Firebase smoke deploy command.

### 4. Manual Firestore Logging Smoke Test

Open the hosted experiment, complete it once, then inspect Firestore for:

```text
collection_sessions/{collection_session_id}/responses/{participant_id}
```

The response document should include:

- `deployment_id`
- `collection_session_id`
- `agent_backend`
- `collection_owner`
- `project_id`
- `experiment_id`
- `run_id`
- `trials`

Then verify CSV export in a browser:

```text
https://auto-psych-2c5da.web.app/results?collection_session_id=<collection_session_id>
```

### Verified Firebase Checkpoint

Verified on June 12, 2026:

- Firebase Hosting served the smoke experiment at `https://auto-psych-2c5da.web.app`.
- The smoke experiment completed two trials in browser.
- `/submit` wrote the participant response to Firestore.
- `/results?collection_session_id=session_subjective_randomness-e1-20260613T000226Z-2bdd60c` exported CSV with two rows.
- Firestore response document included `deployment_id`, `collection_session_id`, `agent_backend`, `collection_owner`, `project_id`, `experiment_id`, `run_id`, Prolific fields set to null, and the full `trials` array.

Current caveat:

- Local Python Firestore deployment/session metadata writes may fail unless Application Default Credentials are set up with `linasmn@stanford.edu` as the active/default Google account. This does not block participant response logging through Firebase Functions.

### 5. Prolific Test Mode

For Prolific test mode, first create
`projects/subjective_randomness/prolific_config.yaml` from the example and set a
test participant email.

```bash
uv run python -m src.pipelines.outer_loop.run \
  --project subjective_randomness \
  --experiment 1 \
  --resume \
  --deploy-only \
  --deploy-target firebase \
  --prolific-mode test \
  --firebase-project auto-psych-2c5da \
  --collection-owner linas
```

## Simulated-Participant Smoke Test (Gemini + Playwright)

`scripts/smoke_firebase_deploy.py` exercises the whole setup end to end with
simulated participants instead of waiting for a Prolific worker. It renders the
real participant template (`templates/jspsych_experiment.html`) over a few H/T
stimuli, deploys to Firebase, then drives N headless-Chromium participants whose
choices are made by **Gemini** (one `ACTION:` per screen, via the
`4_collect_steering` prompt). Each participant's `onFinish` POSTs to `/submit`;
the script then confirms `/results` returns their rows.

Validate the artifacts first without any credentials (renders + stages, no
deploy, no collection):

```bash
uv run python scripts/smoke_firebase_deploy.py --deploy-target dry-run
```

Full live smoke (deploys to production Hosting/Functions and writes to
production Firestore — needs `GOOGLE_API_KEY` and `firebase login`):

```bash
export GOOGLE_API_KEY=...
uv run python scripts/smoke_firebase_deploy.py \
  --confirm-production --n-participants 2 --n-stimuli 4
```

It refuses the live path without `--confirm-production`. Simulated participants
are written under `collection_sessions/<collection_session_id>/responses/*`
(the session id and the participant-id list are printed); delete that session in
the Firebase console afterward if you do not want the smoke data.

## Manual Acceptance

- [ ] Firebase deploy succeeds.
- [ ] Hosted experiment loads.
- [ ] One test participant completes the experiment.
- [ ] Firestore has `collection_sessions/{collection_session_id}/responses/{participant_id}`.
- [ ] The response document includes `deployment_id`, `collection_session_id`, `agent_backend`, `collection_owner`, `project_id`, and trial data.
- [ ] `/results?collection_session_id=...` returns CSV.
- [ ] The outer-loop collect step can write `experiment/data/responses.csv`.
