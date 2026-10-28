# Context

- Candidate directory (write your three files here): `/scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/round_1/candidate_1_retry_1`
- Inner-loop round 1 of 1.

## The paradigms of this chain

Your theory is fitted to all of them at once; the newest is last. The data files are one JSON object per line (`objects`: one row of 0/1 features per object; `heard`: the index of the word heard, `null` on a prior trial; `response` in its paradigm's form). You may analyse them from the shell; your theory file may not read them.

### E: 6703 trials, people click the object the speaker means (the listener)
- link `choice`; data `/scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/data/E.jsonl`
- the reference paradigm: its link's `temperature` = 1 (not fitted); later paradigms' are fitted relative to it

### exp1: 4000 trials, people split points over the objects by how likely the speaker means each (the listener)
- link `allocation` (total_points 10); data `/scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/data/exp1.jsonl`
- Listener 10-point probability allocation (betting) across referents. The prompt: "The speaker said {message}. Split 10 points across the objects to show how likely each one is to be the object the speaker meant."

## The admission limit

The loop's fit of your theory to all the data must finish within **44 minutes** (twice the median fit time of the theories this loop started from), or it is rejected as too slow.

## Self-check

`/scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/venv/bin/python -m src.rsa_open.loop.check_candidate /scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/round_1/candidate_1_retry_1 --corpus /scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/data/check_corpus.json --full-corpus /scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/data/corpus.json --limit-minutes 44`
It compiles your theory and fits it briefly on every paradigm, the literature (E) on a fixed 20% of its displays, in a few minutes; the loop's admission fit uses all the data. It reports how hard the posterior was to sample (steps per draw, step size, and any pair of parameters correlated above 0.85) and projects the admission fit's time from the short fit. Give the shell tool a timeout of 1200000 milliseconds for it. Run it once per change.

- You have no internet access: shell commands cannot open network connections and
  the web tools are disabled. Everything you need is in this repository, your
  candidate directory and the responses file; do not try to install packages.
- Write and test your files **in your candidate directory only**.
- `critiques.md` is a posterior-predictive critique of the current **best** model: the test statistics on which it significantly fails to reproduce the data, each with the direction of the discrepancy and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not confirmatory tests (several are checked per round), so prefer a discrepancy that survives the FDR (`q <= alpha`). Use the strongest such discrepancy to motivate a single mechanism that would close that gap.
