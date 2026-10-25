# Critique context

**Incumbent (best) theory:** `feature_uncertainty_listener_2`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment3/loop/models/feature_uncertainty_listener_2.py`
**Its hypothesis:** Listeners interpret referring expressions under referential feature uncertainty, being unsure which visual feature of a candidate referent the speaker intended to pick out. When evaluating literal semantics, listeners weight an object's compatibility by the probability that the speaker chose the heard feature from that object's feature bundle, naturally favoring minimal referents whose features are exhaustively described over cluttered multi-feature competitors. Communicative speakers select expressions expecting this feature-attribution inference, while spontaneous visual attention on uninformative displays defaults to contextually unique singletons.

**The experiment critiqued:** `exp3`, the newest one (the loop selects theories on it).
Forced-choice referent selection across 2-6 objects and 2-4 features with informative and prior queries and synonyms, testing whether reference theories fitted to graded allocation and single-object ratings generalize to discrete winner-take-all referent decisions across large competitor sets. The prompt: "The speaker says: {message}. Which object is the speaker's target?"
Its link (response process): `choice`.

**The data your statistics receive:** 5600 trials (rows) over 56 distinct scenes
(`objects` x `heard`), as in `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment3/loop/data/exp3.jsonl`, with these columns: `trial, unit, source, objects, heard, choice`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `heard`: the index of the feature whose word was heard (NaN when no word was heard).
- `unit`: the designed trial the row belongs to.
- **The response:** `choice`: the index of the object picked.

**Your job:** propose **8** test statistics. Write each to
`/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment3/loop/round_2/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior through this experiment's link (only `choice` differs between them),
and flags a statistic as a **significant discrepancy** when its two-sided p ≤ 0.05
(with a Benjamini–Hochberg q beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 5600 rows).
Compute scene properties once per distinct scene and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
