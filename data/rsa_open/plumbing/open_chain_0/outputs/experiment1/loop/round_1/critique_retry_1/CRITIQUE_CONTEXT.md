# Critique context

**Incumbent (best) theory:** `rsa_l2_singleton_feat_color_valence_l0`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/models/rsa_l2_singleton_feat_color_valence_l0.py`
**Its hypothesis:** RSA pragmatic listener at depth 2 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_singleton_distinct_feat_color_l0 by incorporating the evaluative valence framing component from polarity_congruence_listener: speakers and listeners reason at depth 2 (inverting a speaker who simulates a depth-1 pragmatic listener) with common knowledge of an object prior combining discrete visual singleton salience (favoring unique objects with no duplicates in the scene), continuous contextual distinctiveness (favoring objects with greater feature contrast relative to the rest of the display), visual feature complexity modulated by evaluative framing valence (favoring feature-rich items under neutral/positive framing but inverting to favor sparse, featureless items when referring to a least favorite object), and visual color contrast (favoring fully colored objects over desaturated grayscale distractors). Crucially, the foundational simulated literal listener L0 operates purely on truth-conditional semantics without perceptual bias, preventing redundant prior compounding across reasoning tiers. On prior trials with no informative word, choices are governed directly by the combined perceptual prior. A lapse parameter mixes in uniform guessing.

**The experiment critiqued:** `exp1`, the newest one (the loop selects theories on it).
Listener 10-point probability allocation (betting) across referents. The prompt: "The speaker said {message}. Split 10 points across the objects to show how likely each one is to be the object the speaker meant."
Its link (response process): `allocation`.

**The data your statistics receive:** 4000 trials (rows) over 48 distinct scenes
(`objects` x `heard`), as in `/scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/data/exp1.jsonl`, with these columns: `trial, unit, source, objects, heard, points`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `heard`: the index of the feature whose word was heard (NaN when no word was heard).
- `unit`: the designed trial the row belongs to.
- **The response:** `points`: a JSON list, the points put on each object.

**Your job:** propose **8** test statistics. Write each to
`/scratch/users/mcfrank/auto-psych/agent_trees/f99f6875513adde3/repo/_runs/chain_0/experiment1/loop/round_1/critique_retry_1/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior through this experiment's link (only `points` differs between them),
and flags a statistic as a **significant discrepancy** when its two-sided p ≤ 0.05
(with a Benjamini–Hochberg q beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 4000 rows).
Compute scene properties once per distinct scene and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
