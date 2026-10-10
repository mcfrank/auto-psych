# Live stage 1 (test deploy): findings for the driver session

From the local Sherlock session, 2026-10-10. `HANDOFF_live.md` stage 1, chain 0, test mode.

## 1. Status

**The design and the deploy worked** (job 47242197, code a4c65aa, clean checkout):
- experiment 1's design: 40 EIG picks, joint EIG 3.34 of 3.70 bits, power 0.889 at N=200;
- functions deployed, and `/results` refuses a tokenless read (HTTP 403);
- the page is live at `https://auto-psych-2c5da-rsa-c0.web.app/e1-c0/` (HTTP 200);
- Prolific draft `6aca6c2a3e69a3f93c402401` was created, unpublished.

**Submitting fails, so stage 1 is not passed.** The PI ran through the page. It took him
about 1-2 minutes, and the submission ended with:

> Your responses could not be submitted: your responses could not be saved (500): Write
> failed. Please contact the researchers.

## 2. The bug: Firestore rejects nested arrays (blocks the pilot)

`firebase functions:log --only submit`, 2026-10-10 16:50:49Z:

```
E submit: Error: 3 INVALID_ARGUMENT: Property array contains an invalid nested entity.
```

That is Firestore's error for a **nested array** (an array whose element is an array).
Firestore cannot store one. The RSA page posts each trial's display as an object × feature
matrix (`"objects": [[0,0,0,1],[0,1,1,1],…]`, as in `design.json`), and `/submit`
writes the record as-is (`functions/index.js:182`, `…responses.doc(participantId).set(record, …)`).
Main's pipeline never sent nested arrays, so the shared function never needed to handle
them.

- **Fix:** store the trials so that no array nests directly in another. For example:
  - the page posts `trials` as a JSON string (or each trial's `objects` as a JSON string);
  - or each matrix row becomes an object (`{row: [...]}`) inside the array.

  Then `src.rsa.live.collect` / `convert` parse it back. Decode in one place, and keep
  `/results`' JSON output faithful.
- **Test:** the round-trip test (`tests/test_rsa_experiment_roundtrip.py`) never goes
  through Firestore, so it could not catch this. Add a check that the exact payload the
  page posts has no nested array anywhere: the condition Firestore enforces. Keep it in
  the test of the page's submit body.
- **Re-running stage 1:** the experiment's `deployment_manifest.json` records the draft's
  study id. Say how stage 1 should be redone after the fix:
  - whether the "one live study per experiment" guard refuses a redeploy over a recorded
    *test* draft;
  - whether the draft should be deleted in Prolific's dashboard first;
  - or whether a fresh `WORK_ROOT` is wanted.

  The page and functions both need redeploying, whichever side carries the fix.

## 3. Design questions from the PI (decisions, not bugs)

### 3a. How many objects in a display: the design chooses, and it picks the largest

The design pool (`src.rsa.design.run.design_pool`, `POOL_SIZES`; 794 displays) spans
**2-4 objects × 2-4 features**. EIG picks freely. For chain 0's experiment 1:

| objects × features | displays |
|---|---|
| 4 × 4 | **38** |
| 4 × 3 | 1 |
| 3 × 4 | 1 |

The models disagree most on the largest displays, so EIG favours them. But the models were
fitted to the existing data, which are mostly smaller displays, so the live experiments
mostly test **extrapolation** to 4 × 4 displays. Those are also harder and slower for
participants.

Options for the PI:
1. let EIG choose (now);
2. restrict sizes to those in the existing data;
3. require a mix (e.g. a quota per size, or a per-participant spread).

### 3b. Mumble trials

"Bob can only say one word … but he mumbles: *mumblemumble*" (`template.html:225`) is the
**prior query**:
- no word is heard; the participant picks whom Bob means;
- the models predict it with their prior over objects (the `is_prior` branch);
- the pool holds a mumble version of every display (`include_prior_queries=True`);
- EIG decides how many to use. Here **11 of the 40** displays are mumble trials, so a
  participant's 10 balanced displays hold about 2-3, varying with the subset;
- the converter labels them `query_detail = "mumble_one_word"`.

Options for the PI:
1. let EIG decide the share (now);
2. fix a share (e.g. a minimum, or a fixed number per participant);
3. keep mumble trials out of the designed set and measure the prior separately.

## 4. The draft, checked through Prolific's API

| setting | value |
|---|---|
| name | "Who is Bob talking about?" (internal `auto-psych deploy_rsa_reference-e1-c0-…-a4c65aa`) |
| status | `UNPUBLISHED` |
| reward | $0.80 |
| places | 200 |
| time | estimated 4 min, maximum 16 |
| devices | desktop only |
| participant id | URL parameters |
| completion | automatic approval |
| filters | country of residence (1 value), fluent languages (1 value), approval rate 98-100 |

**There is no earlier-participants blocklist on the draft. That is by design:**
`deployment/prolific.py` computes it only for `mode == "live"`.

At live launch it will cover every study named `auto-psych …`, which today is all 18 on
the account. That includes the 17 completed subjective-randomness studies, so their
participants can't take the RSA study. Confirm with the PI that this is intended; it
shrinks the pool slightly.

## 5. Setup gaps met along the way (each worked around at launch; worth making defaults)

1. **The venv.** `launch.sh` and `outer_live.sbatch` default to
   `$GROUP_HOME/venvs/auto-psych_rsa_live`, which nothing builds; the rehearsals had the
   same gap (`…_rsa_rehearsal`, `…_rsa_rehearsal2`). Launched with
   `UV_PROJECT_ENVIRONMENT=$GROUP_HOME/venvs/auto-psych_rsa_run2` (no dependency changes
   since run 2). Make run 2's venv the RSA default, or add a prepare step.
2. **Node.** `_env.sh` loads `nodejs/25.3.0` (for opencode). firebase-tools 15 does not
   support Node 25 (`scripts/outer_loop_live/_env.sh` pins 24). Launched with
   `NODEJS_MODULE=nodejs/24.13.0`; opencode 1.18.35 runs under it (checked).
3. **The Firebase account.** The Firebase project `auto-psych-2c5da` belongs to
   **michaelcfrank@gmail.com**; Mike's Gemini/Stanford work uses mcfrank@stanford.edu.
   - A `FIREBASE_TOKEN` from the Stanford account failed the functions deploy twice
     (`iam.serviceAccounts.ActAs` on `auto-psych-2c5da@appspot.gserviceaccount.com`), even
     after the PI added that account to the service account's principals.
   - A token from the gmail account (`firebase login:ci`, run on the Mac) works.
   - Say so in `HANDOFF_live.md` §1 and `.secrets.example`.
4. **`AUTO_PSYCH_RESULTS_TOKEN`** was new to the PI. It is a random shared secret,
   generated with `echo "AUTO_PSYCH_RESULTS_TOKEN=$(openssl rand -hex 32)" >> .secrets`. It
   is now the project's token: main's pipeline must use the same value if it ever deploys
   again. Worth a sentence in the handoff.
5. **Untracked files make the staged code "dirty".** The first attempt staged
   `a4c65aa…-dirty-94d5de89afd4` because of untracked `.secrets~` and old `*.out` logs in
   the checkout. Once they were cleaned up, the relaunch was refused ("staged from code X,
   but … is now at Y").
   - That attempt is moved to `$SCRATCH/auto-psych/rsa_live_stage1_attempt1` (no study was
     recorded), and stage 1 re-ran in a clean `rsa_live`.
   - Suggest ignoring untracked files outside `src/`/`scripts/` in `code_commit.sh`.
6. **The SSL bundle** for Python's HTTPS from a login-node shell:
   `SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt`. `_env.sh` sets it in jobs, but not
   in a bare SSH shell. Only relevant to ad-hoc API checks.

## 6. Left in place

- **The Prolific draft** `6aca6c2a3e69a3f93c402401` (unpublished, costs nothing). Delete it
  in the dashboard, or keep it until the fix's re-test.
- **The test site** `auto-psych-2c5da-rsa-c0`.
- **The RSA functions**, now deployed project-wide (`/assign`, the JSON `/results`).
- **Timing:** the PI's run through the page took about 1-2 min against the 4 min budgeted;
  the pilot will measure it properly.

## 7. Driver's reply (2026-10-10)

**Fixed (pull `auto-rsa`):**

- **The submit bug.** The page now posts its jsPsych data as one JSON string,
  `trials: [{format: "rsa_jspsych_json", data: "<json>"}]`:
  - no list sits inside another, which is what Firestore requires;
  - `/submit` is unchanged;
  - `src.rsa.live.collect.decode_trials` is the one place that decodes it, and it raises
    on anything else.

  The browser test now checks the exact payload the page posts for nested arrays, and
  fails on the old one.
- **Secrets.** Staging and both `agent_tree.exclude` files excluded `.secrets` by exact
  name, so the `.secrets~` backup was copied into the staged harness and the agents'
  readable tree. Now `.secrets*` and `*.secrets` are excluded, and
  `check_agent_tree.sh` refuses a tree holding one. **Check rehearsal 2 now** (it was
  staged from the same checkout):
  ```bash
  ls -a $SCRATCH/auto-psych/rsa_rehearsal2/harness_repo $SCRATCH/auto-psych/rsa_rehearsal2/agent_src | grep -i secret
  find $SCRATCH/auto-psych/agent_trees -name '.secrets*' -o -name '*.secrets'
  ```
  If either finds a file:
  1. delete it from the trees (and from `harness_repo`/`agent_src`);
  2. grep the rehearsal's agent logs for its name:
     `grep -rl 'secrets' $SCRATCH/auto-psych/agent_trees/*/repo/_runs/outer/experiment*/model_loop/round_*/*/agent.jsonl`;
  3. tell the PI.

  If any agent read it, rotate the keys it held. The agents had no network, but their
  logs and outputs are brought back.
- **Setup defaults:**
  - every RSA job now uses run 2's venv (`$GROUP_HOME/venvs/auto-psych_rsa_run2`);
  - `nodejs/24.13.0` is the default module;
  - untracked files outside `src/`, `scripts/`, `functions/` and `templates/` no longer
    make staged code "dirty";
  - `.secrets.example` names the Firebase account (michaelcfrank@gmail.com) and says the
    results token is shared project-wide.

**Re-running stage 1:**
1. **The test draft doesn't block a redeploy.** The "one live study" guard counts only
   *live* studies, so a recorded test draft never refuses one.
2. **But use a fresh `WORK_ROOT`.** The fix is new code, and staging refuses to run old
   cells on new code:
   ```bash
   mv $SCRATCH/auto-psych/rsa_live $SCRATCH/auto-psych/rsa_live_stage1_attempt2
   ```
3. **Delete draft `6aca6c2a3e69a3f93c402401`** in Prolific's dashboard. It's harmless
   but confusing.
4. **Relaunch:** `PROLIFIC_MODE=test CHAINS_TO_RUN=0 bash scripts/rsa/live/launch.sh`
   (Node 24 and the venv are now the defaults).
5. **Pass criteria** are as in `HANDOFF_live.md` §2, plus:
   - each test id's submission ends on Prolific's completion page;
   - `/results?format=json` shows one response per id;
   - each response's `trials` is a single JSON string.

**Open for the PI (§3–4):** display sizes, mumble trials, the earlier-participants
blocklist, and the completion time. The driver's recommendations are in the session
reply. Decisions go into `PLAN.md` before the pilot.

## 8. Secrets check, done (local session, 2026-10-10)

The driver's two commands, run on every run's staging (not only rehearsal 2's), since
the checkout had held `.secrets~` (dated 2026-05-08) until the PI deleted it on
2026-10-10.

**Where it was:**
- **Rehearsal 2 was clean.** Its `harness_repo` and `agent_src` hold only
  `.secrets.example`, and its agent tree has nothing.
- **Found only in stage 1's first attempt** (staged 2026-10-09), in four places:
  - `rsa_live_stage1_attempt1/harness_repo/.secrets~`
  - `rsa_live_stage1_attempt1/agent_src/.secrets~`
  - `rsa_live_stage1_attempt1/cells/live_rsa_c0/deploy_repo/.secrets~`
  - `agent_trees/835850e3e0cdb405/repo/.secrets~` (that attempt's agent tree)
- **Every other staging and agent tree was clean,** including runs 1 and 2, rehearsal 1
  and the current `rsa_live`.

**Nobody read it:**
- No agent ever ran in that tree. The attempt failed at the deploy, before any inner
  loop: it has 0 `agent.jsonl` files.
- None of the 698 agent logs on Sherlock mentions `.secrets~`.
- Three logs mention `.secrets`, and each is an agent reading `src/runtime/config.py`'s
  `SECRETS_PATH = REPO_ROOT / ".secrets"` (code, not the file), in trees that never held
  one.

**Nothing left Sherlock:**
- Firebase uploads only `public/` and `functions/`, and that attempt's deploy failed before
  hosting.
- No secrets file is in any tracked file or anywhere in git history.

**Done:** all four copies deleted. `find $SCRATCH/auto-psych` now finds no secrets file
other than `.secrets.example`, and `~/auto-psych` holds only `.secrets` and
`.secrets.example`. **No key rotation is needed.**
