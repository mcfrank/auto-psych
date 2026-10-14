# Critique context

**Incumbent (best) model:** `bounded_capacity_redundancy_l2`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/94d7a3a1354bcf6e/repo/_runs/outer/experiment1/model_loop/models/bounded_capacity_redundancy_l2.py`
**Its hypothesis:** Refining bounded_capacity_listener_l2, we incorporate contextual feature redundancy discounting on speaker alternatives from feature_redundancy_discounted_l2 into the communicative likelihood. When interpreting referring expressions, capacity-limited listeners reason about speakers who discount the availability of redundant alternative utterances based on their contextual feature overlap across the scene, preventing synonymous or co-occurring attributes from repeatedly compounding the preemption penalty against shared labels. This directly resolves the source model's severe underprediction of competitor choices when competitor referents possess redundant distinguishing features.

Differences from bounded_capacity_listener_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_redundancy (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: None.
- Speaker utterance utility / probability: In S1 and S2, the speaker's candidate utterances are weighted by contextual feature redundancy discounting (vec(alt_weight, u)), where alt_weight scales with contextual pairwise Jaccard feature overlap (jnp.exp(-w_redundancy * jnp.log(jnp.maximum(redundancy, 1.0)))).
- All other terms: Bounded capacity likelihood exponentiation (kappa) in L1 and L2, referent prior (singleton salience w_singleton, contextual distinctiveness w_distinct, feature count w_features, featureless baseline salience w_empty, and color contrast w_color), literal semantics at L0, depth-2 RSA recursion, and uniform lapse mixing remain identical to bounded_capacity_listener_l2.

**The data your statistics receive:** 2314 trials (rows) over 52
distinct displays (`objects` × `utterance`), from auto_psych. The responses CSV
(`/scratch/users/mcfrank/auto-psych/agent_trees/94d7a3a1354bcf6e/repo/_runs/outer/experiment1/model_loop/responses.csv`) holds more columns and rows; your statistic receives only
the included forced-choice trials from `source == "auto_psych"`: the trials the loop selects models on (the live experiments'). The published experiments' trials are in the CSV too, but are not critiqued here and these columns: `source, experiment, condition, participant_id, trial_index, item, feature_names, objects, object_roles, messages, query, utterance, framing, familiarization, grayscale, choice`.
- `objects`: a JSON list of 0/1 feature lists, one per object.
- `query`: `utterance` or `prior` (a mumble trial: no word heard, `utterance` is NaN).
- `utterance`: the index of the heard feature.
- `object_roles`: where known, a JSON list naming each object's role in the original
  design (e.g. target, logical, foil).
- `choice`: the clicked object's index. Identical objects are one choice, recorded
  as the first one's index, and so is every model's simulated choice.
- **Scope: plain displays.** Every trial in the responses and every display the next experiments can show is plain: valence 0 (no favourite / least favourite framing), no familiarization and no greyscale. Mechanisms that act through those fields cannot be tested or rewarded here; spend your proposal on how listeners reason about the words and the display.
- **What the next experiments show** (PI 2026-10-10). The displays are chosen by expected information over the current models, within fixed quotas (of 40 displays, at least 6 with two objects, 10 with three, 8 mumble trials and 20 with a word); the rest go where the models disagree most, which is often the largest displays (4 x 4). Sizes run from 2 objects x 2 features to 4 x 4, and two features may be on exactly the same objects (e.g. two objects sharing three features, one of them with a fourth): to a participant these are different things on the objects, and to a speaker two words, either of which could be said, so a redundant feature changes what a speaker who weighs her alternatives would say. The existing data are mostly 3-object displays without redundant features, so a 4 x 4 display, a two-object display or a redundant feature asks a model to extrapolate: a mechanism that only fits the familiar displays, or behaves oddly as displays grow, will be found out there. Make your model's predictions sensible across display sizes and feature overlaps, and say in your hypothesis how it scales.
- **The speaker's alternatives are yours to propose.** Which utterances a speaker considers is part of a hypothesis: for instance, saying nothing, or a word for a feature no object has, as a way out when every available word is ambiguous. Such a speaker uses an ambiguous word less often, which changes what a listener infers from one. (A display never shows a feature no object has, and the word heard is always true of some object.)
- **Mumble trials** can be part of any experiment (they are in the pool the displays are chosen from, and are chosen where the models' priors disagree): the speaker's word is not heard (`query` is `prior`, `is_prior` in the model's context), and the click shows which object the listener expected the speaker to mean before any word: the prior over referents (salience, distinctiveness, what speakers tend to talk about). A model's prior is half of its account of the word trials too, so it is worth getting right; the data have such trials (pragmods, Sikos 2021) to learn it from.

**Your job:** propose **8** test statistics. Write each to
`/scratch/users/mcfrank/auto-psych/agent_trees/94d7a3a1354bcf6e/repo/_runs/outer/experiment1/model_loop/round_5/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior (only `choice` differs between them), and flags a statistic as a
**significant discrepancy** when its two-sided p ≤ 0.05 (with a Benjamini–Hochberg q
beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 2314 rows).
Compute display properties once per distinct display and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
