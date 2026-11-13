# Handoff: stage 1, open chain 1 and fixed chain 1 (2026-11-13)

For the PI and the Sherlock session. Written by Claude Opus 5.5; not reviewed by a person.

Registered: https://osf.io/wurh5 (`PREREG_V2.md`, Addendum A), tag `rsa-open-prereg-v2`, commit
`b32e732b74d99d5f3dd85a9374d7d2e6b44beab8`. Stage 1 is **open chain 1 and fixed chain 1, run together**, live,
3 experiments each (3 rounds of 6 theorists), judged by the registered gate (`PREREG_V2.md` §4). On a pass they
count as chain 1 of each arm and chains 2-4 of both arms follow.

**A new work root.** The default `$SCRATCH/auto-psych/rsa_open_live` holds the pilot, whose open chain has the
same cell name (`open_chain_0`). Use **`WORK_ROOT=$SCRATCH/auto-psych/rsa_open_v2`** for every command below and
for all of stage 2. A work root runs on the code it first staged, so the checkout is at the tag before anything
runs there. The new chains' blocklists cover only their own earlier studies, so the pilot's participants may take
part, as the registration says (§4a-b).

**How the jobs run** (the tag's defaults): one launch per chain, as one job of 12 cores, 72 GB and 4 days on
`mcfrank` that runs its own fits and every experiment, continued by a new job only if its time runs short. These
launches are the first live use of step mode (chain 1 ran the older ladder); the test deploys below exercise it
without spending anything. Two chains need 24 cores and 144 GB at once: check the node has room
(`sinfo -p mcfrank -N -o "%N %C %m %e"`); if one has to queue, the arms are no longer concurrent, and that is
worth knowing before the live launch.

## 1. The checkout at the tag

```bash
cd ~/auto-psych && git fetch origin --tags && git checkout rsa-open-prereg-v2
git status --short   # must print nothing: the live launcher refuses a dirty checkout or other code
export WORK_ROOT=$SCRATCH/auto-psych/rsa_open_v2 && mkdir -p $WORK_ROOT/logs
```

## 2. Test deploys, both arms (no money)

```bash
PROLIFIC_MODE=test ARM=open  CHAINS=0 bash scripts/rsa/live/open_launch.sh
PROLIFIC_MODE=test ARM=fixed CHAINS=0 bash scripts/rsa/live/open_launch.sh
```

Each job designs experiment 1 (open: the Gemini 4 designer, the harness's checks, the simulated pilot, 2-3 h; fixed:
the frozen spec through the same checks, minutes), deploys its page and a Prolific draft, and stops ("test deploy
done"). If the open designer fails both cycles, the chain restarts once from the same seeds (§4a): report it first.

**Report** (Sherlock session): each page URL and draft id; the open paradigm (`design/frozen/record.json`: response
type, domains, trials, N, pay and minutes, `submitted_by`, the pilot's numbers); the designer's spend; and anything
in either log that step mode did differently from chain 1's ladder. The run trees are under
`$SCRATCH/auto-psych/agent_trees/$(cat $WORK_ROOT/cells/<arm>_chain_0/agent_tree_id)/repo/_runs/chain_0`.

## 3. The PI's review of the open chain's first paradigm

Open both pages (`?PROLIFIC_PID=test-<you>`; it really submits, and a test row never matches a live study) and both
drafts in Prolific's dashboard. For the open paradigm check the materials and instructions, at least 3 picture
domains, the length, the enjoyment question, and the pay and minutes on the listing. Then:

```bash
RUN=$SCRATCH/auto-psych/agent_trees/$(cat $WORK_ROOT/cells/open_chain_0/agent_tree_id)/repo/_runs/chain_0
python -m src.rsa_open.review --frozen $RUN/experiment1/design/frozen --by mcfrank   # or --reject "<why>"
```

## 4. The live launch, both arms (the PI's go-ahead; about $1,600 of participants over the 6 experiments)

**Top up Prolific first** to at least $1,600 plus fees headroom: a publish the balance cannot cover fails after its
draft is recorded. Confirm live recruitment in a copy outside the checkout (which must stay clean), then launch the
open chain and the fixed chain one after the other:

```bash
mkdir -p $WORK_ROOT/private
sed 's/^confirm_live_recruitment: false/confirm_live_recruitment: true/' scripts/rsa/live/open_live.yaml \
  > $WORK_ROOT/private/open_live_confirmed.yaml
for ARM in open fixed; do
  CONFIG=$WORK_ROOT/private/open_live_confirmed.yaml CLEAR_STOP=1 PROLIFIC_MODE=live ARM=$ARM CHAINS=0 \
    bash scripts/rsa/live/open_launch.sh
done
```

Each launch checks the code is the tag, prints the cost and asks for "yes", then submits one job for the chain's
three experiments. Each experiment: design (experiment 1 already frozen), collection (up to 24 h), the loop. No
later review is needed. The job mails when the chain finishes or stops for a person.

**While a study fills:** `python -m src.rsa_open.monitor --exp-dir $RUN/experiment<n>` (exclusions, comprehension,
catch, same-answer flags, minutes, pay per hour at the median). `python -m src.rsa_open.status` shows both chains.

**Report after each experiment** (Sherlock session): `data/participants.json`, `data/timing.json`,
`data/experience.json` (aggregates), the loop's outcome (`loop/export.json`; admitted, too slow, not converged;
the incumbent and `fit_best` with their parameters), the experiment's running time against the gate's 48 h, and the
spend.

## 5. The gate, after both chains' experiment 3

```bash
python -m src.rsa_open.gate --run-dir $WORK_ROOT/cells/open_chain_0/outputs $WORK_ROOT/cells/fixed_chain_0/outputs
# (or the run trees under agent_trees, as in §2); --json for the record
```

The gate's outcome and every item's value go in a report; on a pass, stage 2's launch (chains 2-4 of both arms,
launch order alternating between arms) is the next handoff.
