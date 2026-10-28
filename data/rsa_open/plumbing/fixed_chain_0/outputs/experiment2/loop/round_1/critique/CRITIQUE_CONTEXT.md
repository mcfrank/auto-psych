# Critique context

**Incumbent (best) theory:** `competitor_confusion_shared_prior`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/f30695355035b5ad/repo/_runs/chain_0/experiment2/loop/models/competitor_confusion_shared_prior.py`
**Its hypothesis:** Pragmatic listener at depth 1 reasoning about a competitor-confusion-averse speaker under a shared visual confusability object prior. Refining competitor_confusion_speaker with one change: rather than applying the baseline visual confusability prior only on uninformative prior trials while assuming a uniform object prior when interpreting heard words, the depth-1 listener and simulated speaker share the baseline visual confusability prior over objects when interpreting heard words as well. Recursion depth remains at depth 1 (where the listener L1 inverts a speaker reasoning about a literal listener L0), the parameter set is unchanged with no parameters added or removed (keeping only speaker rationality alpha and confusion aversion w_confusion), and all other terms (exponential feature-distance similarity, utterance-specific competitor confusion penalty, and prior-trial choice probabilities) remain identical.

**The experiment critiqued:** `exp2`, the newest one (the loop selects theories on it).
The closed campaign's reference game: participants see a few pictures, Bob says one word or mumbles, and they click the picture they think he means. The prompt: "Bob can only say one word to communicate with you and he says {message}. Click on the one you think Bob is talking about."
Its link (response process): `choice`.

**The data your statistics receive:** 5200 trials (rows) over 52 distinct scenes
(`objects` x `heard`), as in `/scratch/users/mcfrank/auto-psych/agent_trees/f30695355035b5ad/repo/_runs/chain_0/experiment2/loop/data/exp2.jsonl`, with these columns: `trial, unit, source, objects, heard, choice`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `heard`: the index of the feature whose word was heard (NaN when no word was heard).
- `unit`: the designed trial the row belongs to.
- **The response:** `choice`: the index of the object picked.

**Your job:** propose **8** test statistics. Write each to
`/scratch/users/mcfrank/auto-psych/agent_trees/f30695355035b5ad/repo/_runs/chain_0/experiment2/loop/round_1/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior through this experiment's link (only `choice` differs between them),
and flags a statistic as a **significant discrepancy** when its two-sided p ≤ 0.05
(with a Benjamini–Hochberg q beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 5200 rows).
Compute scene properties once per distinct scene and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
