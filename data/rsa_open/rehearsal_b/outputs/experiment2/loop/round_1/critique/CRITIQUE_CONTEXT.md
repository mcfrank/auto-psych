# Critique context

**Incumbent (best) theory:** `rsa_l1_singleton_feat_color_valence_l0`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment2/loop/models/rsa_l1_singleton_feat_color_valence_l0.py`
**Its hypothesis:** RSA pragmatic listener at depth 1 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_singleton_feat_color_valence_l0 by reducing pragmatic recursion depth from depth 2 to depth 1: listeners interpret referring expressions by inverting an informative speaker who evaluates literal semantics, rather than engaging in higher-order recursive mental simulation. This bounded depth eliminates over-implicature in scalar implicature and prevents multiplicative prior compounding across reasoning tiers, preserving appropriate probability on multi-feature competitors while maintaining common knowledge of visual singleton salience, continuous contextual distinctiveness, valence-modulated feature complexity, and color contrast. Differences from rsa_l2_singleton_feat_color_valence_l0: recursion depth is reduced from depth 2 to depth 1 (listener calls L1 instead of L2), the depth-2 listener memo L2 is removed, and all parameters (alpha, w_singleton, w_distinct, w_features, w_valence, w_color) are preserved without addition or removal.

**The experiment critiqued:** `exp2`, the newest one (the loop selects theories on it).
7-point Likert likelihood ratings of a single probed candidate referent across 2-5 objects and 2-4 features with informative and prior queries and synonyms, testing whether reference theories generalize from multi-object allocation and forced choice to targeted single-referent evaluation. The prompt: "The speaker says: {message}. How likely is it that the highlighted object is the speaker's target?"
Its link (response process): `rating`.

**The data your statistics receive:** 4000 trials (rows) over 45 distinct scenes
(`objects` x `heard`), as in `/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment2/loop/data/exp2.jsonl`, with these columns: `trial, unit, source, objects, heard, probe, rating`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `heard`: the index of the feature whose word was heard (NaN when no word was heard).
- `unit`: the designed trial the row belongs to.
- `probe`: the trial's design (as presented, the same in every simulated dataset).
- **The response:** `rating`: the rating given to the probed object `probe` (1 = lowest).

**Your job:** propose **8** test statistics. Write each to
`/scratch/users/mcfrank/auto-psych/agent_trees/e6344f63e5fb0046/repo/_runs/chain_0/experiment2/loop/round_1/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior through this experiment's link (only `rating` differs between them),
and flags a statistic as a **significant discrepancy** when its two-sided p ≤ 0.05
(with a Benjamini–Hochberg q beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 4000 rows).
Compute scene properties once per distinct scene and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
