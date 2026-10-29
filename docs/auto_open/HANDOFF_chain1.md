# Handoff: open chain 1, the feasibility chain (2026-10-29)

Registered: https://osf.io/78t69, tag `rsa-open-prereg` (commit `5dc1e170a7cc`; `PREREG_DRAFT.md`
Addendum A). Open chain 1 is the code's `chain_0` (`CHAINS=0`; seeds `data/rsa_open/seeds/chain_0`,
the closed campaign's chain 0 starting set). It runs **3 experiments** (`N_EXPERIMENTS=3`), which the
gate judges (`PREREG_DRAFT.md` §4); its 4th runs only if the gate passes and stage 2 keeps 4
experiments per chain, by a registered amendment.

**The campaign's work root is new:** `$SCRATCH/auto-psych/rsa_open_live` (the launcher's default),
never the smoke root. A work root runs on the code it first staged, so the checkout must be at the
tag **before the test deploy**, and stay there.

## 1. The checkout at the tag

```bash
cd ~/auto-psych && git fetch origin --tags && git checkout rsa-open-prereg
git status --short   # must print nothing: the live launcher refuses a dirty checkout or other code
```

## 2. The test deploy (no money)

```bash
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_live
PROLIFIC_MODE=test ARM=open CHAINS=0 N_EXPERIMENTS=3 LINKS=1 bash scripts/rsa/live/open_launch.sh
```

The job designs experiment 1 (the Gemini 4 designer, the harness's checks and the simulated pilot;
2–3 h, up to two cycles), deploys its page and a Prolific draft, and stops with a STOP record
("test deploy done"). If the designer fails both cycles, the chain is restarted once from the same
seeds with a fresh designer (`PREREG_DRAFT.md` §4a): report it first.

**Report:** the log's page URL and draft id, the frozen paradigm (`design/frozen/record.json`:
response type, domains, trials, N, pay and minutes, `submitted_by`; the pilot's numbers), the
designer's spend. The run tree is
`$SCRATCH/auto-psych/agent_trees/$(cat $WORK_ROOT/cells/open_chain_0/agent_tree_id)/repo/_runs/chain_0`.

## 3. The PI's review

Open the page (`?PROLIFIC_PID=test-<you>`; it really submits, and a test row never matches the live
study) and the draft in Prolific's dashboard. Check: the materials and instructions, at least 3
picture domains, the length, the end-of-study enjoyment question (its first real use), the pay
and minutes on the listing. Then:

```bash
RUN=$SCRATCH/auto-psych/agent_trees/$(cat $WORK_ROOT/cells/open_chain_0/agent_tree_id)/repo/_runs/chain_0
python -m src.rsa_open.review --frozen $RUN/experiment1/design/frozen --by mcfrank   # or --reject "<why>"
```

## 4. The live launch (the PI's go-ahead; about $800 of participants over 3 experiments)

**Top up Prolific first:** after run D2 the balance was $48.67, and each experiment spends up to
$266.67; a publish the balance cannot cover fails after the draft is recorded. Confirm live
recruitment in a copy outside the checkout (it must stay clean):

```bash
mkdir -p $WORK_ROOT/private
sed 's/^confirm_live_recruitment: false/confirm_live_recruitment: true/' scripts/rsa/live/open_live.yaml \
  > $WORK_ROOT/private/open_live_confirmed.yaml
CONFIG=$WORK_ROOT/private/open_live_confirmed.yaml CLEAR_STOP=1 PROLIFIC_MODE=live ARM=open CHAINS=0 \
  N_EXPERIMENTS=3 bash scripts/rsa/live/open_launch.sh
```

The launcher checks the code is the tag, prints the cost, and asks for "yes". It queues 4 jobs of
48 h (each resumes the chain where the last stopped). Each experiment: design (except experiment 1,
already frozen), collection (up to 24 h), the loop (about 20 h). Later experiments need no review.

**While a study fills:** `python -m src.rsa_open.monitor --exp-dir $RUN/experiment<n>` (exclusions,
comprehension, catch, same-answer flags, minutes, pay per hour at the median: under $9 it warns;
Prolific pauses a study under $8, and then the reward is raised in the dashboard).

**Report after each experiment:** `data/participants.json` (aggregates), `data/timing.json`,
`data/experience.json`, the loop's outcome (`loop/export.json`, admitted / too slow / not
converged), the stage times, and the spend.

## 5. The gate, after experiment 3

```bash
python -m src.rsa_open.gate --run-dir $RUN          # the registered script; --json for the record
```

Report its output verbatim. Nothing in stage 2 starts before the PI's decision and its registered
amendment.

## Bring back (to `data/rsa_open/live/open_chain_0/`)

Each experiment's `design/frozen/` (no page files), `data/participants.json`, `timing.json`,
`experience.json`, `loop/history.json`, `export.json`, `newest_block_cv.json`, `models/` and
`attempted_hypotheses.jsonl`; the chain's `stage_times.jsonl` and `token_usage_summary.json`; the
gate's JSON. **Never:** `rows.jsonl`, `block.json`, the corpus, `private/`, `raw_collected/` or the
campaign's id map (Prolific ids, trial-level data and comments stay on Sherlock).
