# Context

- Candidate directory (write your three files here): `/scratch/users/mcfrank/auto-psych/agent_trees/8d59918afaad7427/repo/_runs/chain_0/experiment1/loop/round_1/candidate_1`
- Inner-loop round 1 of 1.

## Your time

This session ends after **60 minutes**; every self-check prints the time left. **Write a complete first version of your three files within 15 minutes**, run the self-check on it, then improve the files in place. Whatever is in your directory when the session ends is what the loop judges; a session that ends with no candidate.py has wasted its slot. Do not spend the session on fitting scripts of your own: the quick fit below answers what they would, in seconds.

## The paradigms of this chain

Your theory is fitted to all of them at once; the newest is last. The data files are one JSON object per line (`objects`: one row of 0/1 features per object; `heard`: the index of the word heard, `null` on a prior trial; `response` in its paradigm's form). You may analyse them from the shell; your theory file may not read them.

### E: 6703 trials, people click the object the speaker means (the listener)
- link `choice`; data `/scratch/users/mcfrank/auto-psych/agent_trees/8d59918afaad7427/repo/_runs/chain_0/experiment1/loop/data/E.jsonl`
- the reference paradigm: its link's `temperature` = 1 (not fitted); later paradigms' are fitted relative to it

### exp1: 5200 trials, people click the object the speaker means (the listener)
- link `choice`; data `/scratch/users/mcfrank/auto-psych/agent_trees/8d59918afaad7427/repo/_runs/chain_0/experiment1/loop/data/exp1.jsonl`
- The closed campaign's reference game: participants see a few pictures, Bob says one word or mumbles, and they click the picture they think he means. The prompt: "Bob can only say one word to communicate with you and he says {message}. Click on the one you think Bob is talking about."

## The admission limit

The loop's fit of your theory to all the data must finish within **33 minutes** (twice the median fit time of the theories this loop started from), or it is rejected as too slow.

## Self-check

`/scratch/users/mcfrank/auto-psych/agent_trees/8d59918afaad7427/venv/bin/python -m src.rsa_open.loop.check_candidate /scratch/users/mcfrank/auto-psych/agent_trees/8d59918afaad7427/repo/_runs/chain_0/experiment1/loop/round_1/candidate_1 --corpus /scratch/users/mcfrank/auto-psych/agent_trees/8d59918afaad7427/repo/_runs/chain_0/experiment1/loop/data/check_corpus.json --full-corpus /scratch/users/mcfrank/auto-psych/agent_trees/8d59918afaad7427/repo/_runs/chain_0/experiment1/loop/data/corpus.json --limit-minutes 33`
It compiles your theory and fits it briefly on every paradigm, the literature (E) on a fixed 20% of its displays, in a few minutes; the loop's admission fit uses all the data. It reports how hard the posterior was to sample (steps per draw, step size, and any pair of parameters correlated above 0.85) and projects the admission fit's time from the short fit. Give the shell tool a timeout of 1200000 milliseconds for it. Run it once per change.

Add `--quick` to the same command for the quick fit instead (seconds, no sampling): the best-fitting values from several random starts, how sharply the data pin each parameter down (a Laplace approximation), which parameters trade off, and whether the starts disagree (more than one optimum). It is for exploring; the loop scores theories with full sampling, never with it.

- You have no internet access: shell commands cannot open network connections and
  the web tools are disabled. Everything you need is in this repository, your
  candidate directory and the responses file; do not try to install packages.
- Write and test your files **in your candidate directory only**.
- `critiques.md` is a posterior-predictive critique of the current **best** model: the test statistics on which it significantly fails to reproduce the data, each with the direction of the discrepancy and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not confirmatory tests (several are checked per round), so prefer a discrepancy that survives the FDR (`q <= alpha`). Use the strongest such discrepancy to motivate a single mechanism that would close that gap.
