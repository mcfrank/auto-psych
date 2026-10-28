# Critique context

**Incumbent (best) theory:** `rsa_l2_singleton_feat_color_valence_l0`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/models/rsa_l2_singleton_feat_color_valence_l0.py`
**Its hypothesis:** RSA pragmatic listener at depth 2 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0. Refining rsa_l2_singleton_distinct_feat_color_l0 by incorporating the evaluative valence framing component from polarity_congruence_listener: speakers and listeners reason at depth 2 (inverting a speaker who simulates a depth-1 pragmatic listener) with common knowledge of an object prior combining discrete visual singleton salience (favoring unique objects with no duplicates in the scene), continuous contextual distinctiveness (favoring objects with greater feature contrast relative to the rest of the display), visual feature complexity modulated by evaluative framing valence (favoring feature-rich items under neutral/positive framing but inverting to favor sparse, featureless items when referring to a least favorite object), and visual color contrast (favoring fully colored objects over desaturated grayscale distractors). Crucially, the foundational simulated literal listener L0 operates purely on truth-conditional semantics without perceptual bias, preventing redundant prior compounding across reasoning tiers. On prior trials with no informative word, choices are governed directly by the combined perceptual prior. A lapse parameter mixes in uniform guessing.

**The experiment critiqued:** `exp1`, the newest one (the loop selects theories on it).
The closed campaign's reference game: participants see a few pictures, Bob says one word or mumbles, and they click the picture they think he means. Then Bob marks one picture, and they rate how likely it is that he meant that one. The prompt: "Bob can only say one word to communicate with you and he says {message}. Click on the one you think Bob is talking about."
Its link (response process): `pair`.

**The data your statistics receive:** 3744 trials (rows) over 49 distinct scenes
(`objects` x `heard`), as in `/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/data/exp1.jsonl`, with these columns: `trial, unit, source, objects, heard, choice, second_probe, second_rating`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `heard`: the index of the feature whose word was heard (NaN when no word was heard).
- `unit`: the designed trial the row belongs to.
- `second_probe`: the trial's design (as presented, the same in every simulated dataset).
- **The response:** `choice`: the index of the object picked; then `second_rating`: the second question's rating of the probed object `second_probe` (1 = lowest).

**Your job:** propose **8** test statistics. Write each to
`/scratch/users/mcfrank/auto-psych/agent_trees/0aa8c466d948a5ea/repo/_runs/chain_0/experiment1/loop/round_1/critique_retry_1/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior through this experiment's link (only `choice`, `second_rating` differ between them),
and flags a statistic as a **significant discrepancy** when its two-sided p ≤ 0.05
(with a Benjamini–Hochberg q beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 3744 rows).
Compute scene properties once per distinct scene and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
