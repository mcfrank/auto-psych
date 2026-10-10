# Handoff: the RSA live campaign, in three stages (test deploy → pilot → campaign)

From the driver session, 2026-10-10, for the local Sherlock session. **Each stage needs
the PI's go-ahead, and stages 2 and 3 spend real money.** Rehearsal 2
(`HANDOFF_rehearsal2.md`) should finish first: it exercises the same loop with simulated
people.

## 0. What is new

**Cloud Functions** (`functions/index.js`, `functions/lists.js`, `firebase.json`; additive):
- `POST /assign` gives each participant of a collection session the next trial list in
  arrival order. A reload gets the same list.
- `GET /results?format=json` returns responses whole. The CSV export knows only main's
  columns and would drop every RSA trial.
- The first RSA deploy publishes these functions **project-wide**. Main's pages don't call
  them. But a later deploy *from main* would remove them again: don't deploy main's
  pipeline while an RSA study is recruiting.

**The live page** (`src/rsa/live/site.py`):
- the deployment's IRB consent gate (`templates/consent.txt`) comes first;
- then the list from `/assign`;
- then 1 practice screen and 12 choices (10 designed + 2 catch);
- then the post to `/submit` and the redirect to Prolific's completion URL.

A link without a participant id stops the page rather than picking a list. A failed post
shows an error and does not redirect.

**Collection** (`src/rsa/live/collect.py`, `OuterRun.collect_live`):
- main's `run_deployment` deploys the page and runs Prolific. It never deploys twice over
  a recorded live study.
- main's Prolific polling (3 h) and pause handle recruitment.
- `/results` JSON is converted by the RSA converter. Participants are numbered `live<k>`
  across a run's experiments; catch-trial exclusion is as in the simulations.
- **Prolific ids live only in the cell's `private/`** (`raw_collected/`,
  `participant_ids.json`). Nothing agents read has them. Never copy `private/` off
  Sherlock.

**Config** (`scripts/rsa/live/rsa_live.yaml`):
- 3 chains × 3 experiments × 200 people, 40 displays;
- $12/h for an estimated 4 minutes: $0.80 reward, about $1.06 with Prolific's fee;
- campaign total about **$1,915**;
- Hosting sites `auto-psych-2c5da-rsa-c0/1/2`.

`confirm_live_recruitment` is committed as `false`.

**Launcher** (`scripts/rsa/live/launch.sh` → `scripts/rsa/slurm/outer_live.sbatch`, one
array task per chain): it checks the config, prints the cost, and for live asks for a
typed `yes`. Each chain deploys from its own copy of the staged code, with provenance
recorded; deploys share a lock.

## 1. Pull

```bash
cd ~/auto-psych && git fetch origin && git merge --ff-only origin/auto-rsa
```

Make sure `.secrets` has `PROLIFIC_API_TOKEN`, `AUTO_PSYCH_RESULTS_TOKEN` and
`FIREBASE_TOKEN` (the job checks; never print them).

**After stage 1's first attempt** (`STAGE1_FINDINGS.md`): re-run in a fresh `WORK_ROOT` (§7 there). Use a `FIREBASE_TOKEN` from the account that owns the Firebase project (michaelcfrank@gmail.com).

## 2. Stage 1: test deploy (no money)

```bash
PROLIFIC_MODE=test CHAINS_TO_RUN=0 bash scripts/rsa/live/launch.sh
```

**What happens:** chain 0 designs experiment 1, deploys its page to
`https://auto-psych-2c5da-rsa-c0.web.app/e1-c0/`, creates a **draft** Prolific study and
stops. Expect about 30 min (the design fits).

**Check, and report:**
- **The deploy log:** functions deployed; `/results` refuses without the token; the site
  is live.
- **The page.** Open `…/e1-c0/?PROLIFIC_PID=TEST1&STUDY_ID=x&SESSION_ID=y` twice (two
  ids). Confirm:
  - consent, then the practice screen, then 12 choices;
  - at the end it redirects to Prolific's completion page;
  - each id gets a different list, and reloading with the same id gives the same list.
- **The draft in Prolific's dashboard:** name, reward, places, device filter, and the
  earlier-participants exclusion.
- **The test submissions:**
  ```bash
  curl -s -H "x-results-token: $AUTO_PSYCH_RESULTS_TOKEN" \
    "https://auto-psych-2c5da-rsa-c0.web.app/results?collection_session_id=<from the manifest>&format=json"
  ```
  This should list them, trials included. Don't paste the token anywhere.

The live run later deploys a fresh session, so test submissions never mix with real data.
The collector also keeps only responses whose study id is the live study's.

## 3. Stage 2: pilot (about 20 people, about $21; PI go-ahead)

Set `confirm_live_recruitment: true` in the config (locally, don't commit), then:

```bash
PROLIFIC_MODE=live PILOT=1 PARTICIPANTS=20 CHAINS_TO_RUN=0 bash scripts/rsa/live/launch.sh
```

This is its own run (`live_rsa_c0_pilot`, pages at `…/e1-c0pilot/`). It stops once the
data are in. **Report:**
- the median completion time on Prolific, against the 4-minute estimate the pay assumes;
- the catch-trial exclusions (`data/participants.json`);
- each designed display's response count, which should be equal ±1 (`responses.csv`,
  `condition`);
- any participant complaints or returns.

## 4. Stage 3: the campaign (about $1,915; PI go-ahead after the pilot)

```bash
PROLIFIC_MODE=live bash scripts/rsa/live/launch.sh
```

Three array tasks, one per chain, each 3 experiments. Each experiment is: a design
(about 30 min), recruitment (up to 3 h; the study is paused if short), the prospective
score, then the inner loop (about 7 h).
- **Owners overflow:** add `-p mcfrank,owners --requeue` only if the node is full. A
  requeued task resumes and never republishes.
- **To stop:** `scancel` stops the job but **not** a published study. Pause it in Prolific's
  dashboard.

**Bring back** (into `data/rsa/live/<cell>/`):
- `outputs/` **without** the trial-level CSVs (`experiment*/data/*.csv`,
  `model_loop/responses.csv`, `.cv/fold_*.csv`);
- from `private/`, **only** `outer_config.json` and `experiment*/prospective.json`
  (claim 2's record). **Never** bring back `raw_collected/` or `participant_ids.json`
  (Prolific ids), or the fit cache.

The trial data stay on Sherlock until we decide on de-identified sharing.

**Report per chain and experiment:**
- `prospective.json` `committed_vs` (claim 2);
- `carry.json`;
- the design's power;
- the inner loop's best model;
- the agent spend (`token_usage_summary.json`);
- the Prolific cost.
