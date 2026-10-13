# Critique context

**Incumbent (best) model:** `focal_bounded_info_gain_listener`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment3/model_loop/models/focal_bounded_info_gain_listener.py`
**Its hypothesis:** Listeners interpret referring expressions under capacity-limited visual attention that restricts counterfactual communicative mentalizing to the active focal set of candidate referents matching the heard word, reasoning about a speaker who balances referent recovery with global communicative information gain under bounded consideration. When evaluating what a speaker would have said, listeners mentally simulate alternative expressions only with respect to distinguishing candidates within this attended focal contrast set rather than tracking distractors already ruled out by the literal word. Consequently, distinguishing features that isolate a competitor within the focal set strongly preempt that competitor, elevating choices of multi-feature pragmatic targets over all-feature competitors in four-object scenes and calibrating choices when targets possess unique alternatives.

Refining bounded_info_gain_outlier_listener by incorporating the focal candidate attention component from focal_candidate_attention_listener: listeners restrict counterfactual communicative mentalizing to the active focal set of candidate referents matching the heard word. When evaluating what a speaker would have said, listeners mentally simulate alternative expressions only with respect to distinguishing candidates within this attended contrast set rather than tracking distractors already ruled out by the literal word. Consequently, distinguishing features that isolate a competitor within the focal candidate set strongly preempt that competitor, elevating choices of multi-feature pragmatic targets over all-feature competitors in four-object displays (four_object_multifeature_target_vs_all_feature_rate and four_object_two_feature_target_vs_three_feature_competitor_rate), elevating target choices when targets possess unique alternatives (unique_alternative_target_choice_rate), and moderating single-feature target choices in three-object displays (three_object_one_feature_target_vs_two_feature_competitor_rate), directly rectifying the primary significant discrepancies identified in the posterior-predictive critique while preserving communicative information gain, bounded speaker consideration, contextual category typicality, and contrastive outlier pop-out.

Differences from bounded_info_gain_outlier_listener:
- Recursion depth remains depth 2 (choice_probs calls L_typical, which inverts a speaker simulating depth-1 listener L1).
- Added parameter: logit_att_leak (prior dist.Normal(-2.0, 1.0)).
- No parameters removed (retains alpha, logit_p_consider, logit_p_base, w_load, w_popout, w_candidate, w_info, logit_p_outlier, w_inhibit_leak, w_typical, w_singleton, w_distinct, w_features, w_valence, w_color, w_confusion, and lapse).
- In choice_probs: computed att_leak = jax.nn.sigmoid(params["logit_att_leak"]), focal candidate mask focal = focal_weights / jnp.sum(focal_weights) where focal_weights = jnp.where(ctx.is_prior > 0, 1.0, jnp.where(real_matches > 0, 1.0, att_leak)), and focal candidate prior focal_prior = focal_prior_weights / jnp.sum(focal_prior_weights) where focal_prior_weights = prior * focal_weights + 1e-6, passing focal and focal_prior to L_typical.
- In L0: modulated literal listener choice weights by focal candidate mask: wpp=at(lex, u, r) * vec(focal, r) + {EPS}.
- In L1 and L_typical: simulated speaker draws referents from focal_prior (speaker: given(r in OBJ, wpp=vec(focal_prior, r))) instead of the scene-wide prior.

**The data your statistics receive:** 7319 trials (rows) over 109
distinct displays (`objects` × `utterance`), from auto_psych. The responses CSV
(`/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment3/model_loop/responses.csv`) holds more columns and rows; your statistic receives only
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
`/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment3/model_loop/round_3/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior (only `choice` differs between them), and flags a statistic as a
**significant discrepancy** when its two-sided p ≤ 0.05 (with a Benjamini–Hochberg q
beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 7319 rows).
Compute display properties once per distinct display and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
