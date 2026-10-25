# Context

- Candidate directory (write your three files here): `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/round_3/candidate_6_retry_1`
- Inner-loop round 3 of 5.

## The paradigms of this chain

Your theory is fitted to all of them at once; the newest is last. The data files are one JSON object per line (`objects`: one row of 0/1 features per object; `heard`: the index of the word heard, `null` on a prior trial; `response` in its paradigm's form). You may analyse them from the shell; your theory file may not read them.

### E: 50587 trials, people click the object the speaker means (the listener)
- link `choice`; data `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/data/E.jsonl`
- the reference paradigm: its link's `temperature` = 1 (not fitted); later paradigms' are fitted relative to it

### exp1: 1800 trials, people split points over the objects by how likely the speaker means each (the listener)
- link `allocation` (total_points 10); data `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/data/exp1.jsonl`
- Graded 10-point belief allocation across 3-4 objects and 2-3 features on informative and prior trials, testing whether theories fitted to forced choice generalize to full referent probability distributions. The prompt: "The speaker says: {message}. Split 10 points across the objects to show how likely each one is to be the speaker's target."

## Self-check

`/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/venv/bin/python -m src.rsa_open.loop.check_candidate /scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/round_3/candidate_6_retry_1 --corpus /scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment1/loop/data/check_corpus.json`
It compiles your theory and fits it briefly on every paradigm, the literature (E) on a fixed 20% of its displays, in a few minutes; the loop's admission fit uses all the data. Give the shell tool a timeout of 1200000 milliseconds for it. Run it once per change.

- You have no internet access: shell commands cannot open network connections and
  the web tools are disabled. Everything you need is in this repository, your
  candidate directory and the responses file; do not try to install packages.
- Write and test your files **in your candidate directory only**.
- `critiques.md` is a posterior-predictive critique of the current **best** model: the test statistics on which it significantly fails to reproduce the data, each with the direction of the discrepancy and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not confirmatory tests (several are checked per round), so prefer a discrepancy that survives the FDR (`q <= alpha`). Use the strongest such discrepancy to motivate a single mechanism that would close that gap.
