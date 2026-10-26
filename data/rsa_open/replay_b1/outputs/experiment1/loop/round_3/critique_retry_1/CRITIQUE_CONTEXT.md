# Critique context

**Incumbent (best) theory:** `contextual_contrast_listener`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/models/contextual_contrast_listener.py`
**Its hypothesis:** Speakers choose referring expressions to maximize contextual contrast against visually similar competitors in the immediate display, rather than simulating recursive mental states. Listeners interpret utterances by inverting this contrastive speaker, expecting heard words to discriminate the target from its nearest context competitors. On uninformative displays without a referring expression, visual singleton salience guides referent expectations.

**The experiment critiqued:** `exp1`, the newest one (the loop selects theories on it).
Graded 10-point belief allocation across 3-4 objects and 2-3 features on informative and prior trials, testing whether theories fitted to forced choice generalize to full referent probability distributions. The prompt: "The speaker says: {message}. Split 10 points across the objects to show how likely each one is to be the speaker's target."
Its link (response process): `allocation`.

**The data your statistics receive:** 1800 trials (rows) over 45 distinct scenes
(`objects` x `heard`), as in `/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/data/exp1.jsonl`, with these columns: `trial, unit, source, objects, heard, points`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `heard`: the index of the feature whose word was heard (NaN when no word was heard).
- `unit`: the designed trial the row belongs to.
- **The response:** `points`: a JSON list, the points put on each object.

**Your job:** propose **8** test statistics. Write each to
`/scratch/users/mcfrank/auto-psych/agent_trees/34f4248e29dbad0a/repo/_runs/chain_0/experiment1/loop/round_3/critique_retry_1/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior through this experiment's link (only `points` differs between them),
and flags a statistic as a **significant discrepancy** when its two-sided p ≤ 0.05
(with a Benjamini–Hochberg q beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 1800 rows).
Compute scene properties once per distinct scene and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
