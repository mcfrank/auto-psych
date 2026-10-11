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
- 3 chains × 3 experiments × 200 people; 15 trials each (13 designed + 2 catch) over 52 displays;
- $0.50 listed at 3 minutes ($10/h; PI 2026-10-11, after the pilot's 2.0-min median);
- campaign total about **$1,200**, experiment 1 of the three chains about $399;
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
stops. Expect about 1.5 h: the fits (about 30 min), then the display selection (about
1 h; the pool doubled on 2026-10-10, and the free design is selected beside it on a
second core).

**Check, and report:**
- **The design** (`experiment1/design/eig.json`): the quotas met (`designs[0].quotas`:
  each `count` ≥ `minimum`), the mix (`kinds`), the power (`power`) beside the free
  design's (`free.power`), and `screened_out` (should be empty). The local run of the same
  design on 2026-10-10 (fits to all human trials) gave power 0.760 ± 0.007 with quotas vs
  0.767 ± 0.007 free, 6 two-object displays vs 1.
- **The deploy log:** functions deployed; `/results` refuses without the token; the site
  is live.
- **The page.** Open `…/e1-c0/?PROLIFIC_PID=TEST1&STUDY_ID=x&SESSION_ID=y` twice (two
  ids). Confirm:
  - consent, then the practice screen, then 12 choices;
  - at the end it redirects to Prolific's completion page;
  - each id gets a different list, and reloading with the same id gives the same list;
  - **the new display kinds look right:** a two-object display, and one where two features
    are on exactly the same objects (both 2×4 displays are like that). Find a list that has
    them in `experiment1/design/trial_lists.json` (`objects` with 2 rows), and use its
    index + 1 as the participant's arrival order (or use more test ids).
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

## 4. Stage 3: the campaign, experiment 1 first (PI go-ahead, 2026-10-11)

**What changed since the pilot** (`CAMPAIGN_LOG.md` #7, 24, 25, 43, 51–53):
- **Trials:** 15 a person (13 designed + 2 catch), over 52 designed displays (50 responses each).
- **Pay:** $0.50, listed at 3 min. Prolific's description now says "About 3 minutes".
- **The critic** scores the live trials only.
- **Admission** rejects a candidate whose hypothesis repeats a model's word for word.

**The code is frozen at `b162c960`.** Later commits on `auto-rsa` change docs only until the campaign ends, unless a bug fix is logged in `CAMPAIGN_LOG.md`. Record the commit you stage (`$WORK_ROOT/code_commit`) in the report.

**1. Pull, and use a fresh work root.** `rsa_live` was staged from older code.

```bash
cd ~/auto-psych && git fetch origin && git merge --ff-only origin/auto-rsa && git log --oneline -1
export WORK_ROOT=$SCRATCH/auto-psych/rsa_campaign
```

**2. Confirm, locally and uncommitted.** Set `confirm_live_recruitment: true` in a copy of
`scripts/rsa/live/rsa_live.yaml`, passed as `CONFIG`, as for the pilot.

**3. Launch experiment 1 of every chain.**

```bash
PROLIFIC_MODE=live N_EXPERIMENTS=1 WORK_ROOT=$WORK_ROOT CONFIG=<the copy> bash scripts/rsa/live/launch.sh
```

The launcher prints "this launch: … 3 studies, ~$399" before the typed yes. There are three
array tasks, one per chain. Each chain's experiment 1 is:
- a design, about 1.5 h (52 displays);
- recruitment, up to 3 h (the study is paused if short);
- the prospective score;
- the inner loop: 5 rounds × 6 slots with a critique before each, about 7–9 h.

Notes:
- **Owners overflow:** add `-p mcfrank,owners --requeue` only if the node is full. A
  requeued task resumes and never republishes.
- **To stop:** `scancel` stops the job but **not** a published study. Pause the study in
  Prolific's dashboard.
- **Held submissions:** check Prolific's dashboard for submissions held for review while
  each study recruits.

**4. After experiment 1 (report, then PI go-ahead).** Resubmit the same command with
`N_EXPERIMENTS=3`:
- each chain continues in place, and experiment 1's stages are kept;
- the staging step refuses to run on code other than the frozen commit;
- experiments 2–3 cost about $800.

**Bring back** (into `data/rsa/live/<cell>/`):
- `outputs/` **without** the trial-level CSVs (`experiment*/data/*.csv`,
  `model_loop/responses.csv`, `.cv/fold_*.csv`);
- from `private/`, **only** `outer_config.json` and `experiment*/prospective.json`
  (claim 2's record). **Never** bring back `raw_collected/` or `participant_ids.json`
  (Prolific ids), or the fit cache.

The trial data stay on Sherlock until we decide on de-identified sharing.

**Report per chain and experiment** (after experiment 1, before the go-ahead for 2–3):
- the frozen commit (`$WORK_ROOT/code_commit`);
- completion times against the 3-min listing, exclusions, held submissions, and complaints or returns;
- the critic's statistics per round (now on the live trials): which were significant, and how
  many candidates cite them;
- `prospective.json` `live_vs` in experiment 1; `committed_vs` from experiment 2 on (claim 2);
- `carry.json`;
- the design's power;
- the inner loop's best model;
- the agent spend (`token_usage_summary.json`);
- the Prolific cost.
