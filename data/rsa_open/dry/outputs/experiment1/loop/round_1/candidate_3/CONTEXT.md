# Context

- Candidate directory (write your three files here): `/scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/repo/_runs/chain_0/experiment1/loop/round_1/candidate_3`
- Inner-loop round 1 of 1.

## The paradigms of this chain

Your theory is fitted to all of them at once; the newest is last. The data files are one JSON object per line (`objects`: one row of 0/1 features per object; `heard`: the index of the word heard, `null` on a prior trial; `response` in its paradigm's form). You may analyse them from the shell; your theory file may not read them.

### E: 6703 trials, people click the object the speaker means (the listener)
- link `choice`; data `/scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/repo/_runs/chain_0/experiment1/loop/data/E.jsonl`
- the reference paradigm: its link's `temperature` = 1 (not fitted); later paradigms' are fitted relative to it

### exp1: 1800 trials, people split points over the objects by how likely the speaker means each (the listener)
- link `allocation` (total_points 10); data `/scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/repo/_runs/chain_0/experiment1/loop/data/exp1.jsonl`
- Graded belief allocation (10-point bet) across candidate referents on 2- to 4-object scenes with and without heard messages. The prompt: "The speaker says: {message}. Split 10 points across the objects to show how likely each one is to be the object the speaker means."

## Self-check

`/scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/venv/bin/python -m src.rsa_open.loop.check_candidate /scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/repo/_runs/chain_0/experiment1/loop/round_1/candidate_3 --corpus /scratch/users/mcfrank/auto-psych/agent_trees/544a76a2b5f3d6aa/repo/_runs/chain_0/experiment1/loop/data/corpus.json`
It takes 3-15 minutes (it compiles and fits your theory on every paradigm). Give the shell tool a timeout of 1200000 milliseconds for it. Run it once per change.

- You have no internet access: shell commands cannot open network connections and
  the web tools are disabled. Everything you need is in this repository, your
  candidate directory and the responses file; do not try to install packages.
- Write and test your files **in your candidate directory only**.
- `critiques.md` is a posterior-predictive critique of the current **best** model: the test statistics on which it significantly fails to reproduce the data, each with the direction of the discrepancy and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not confirmatory tests (several are checked per round), so prefer a discrepancy that survives the FDR (`q <= alpha`). Use the strongest such discrepancy to motivate a single mechanism that would close that gap.
