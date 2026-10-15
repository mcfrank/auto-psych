# Critique context

**Incumbent (best) model:** `feature_binding_salience_lapse_l2`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/94d7a3a1354bcf6e/repo/_runs/outer/experiment3/model_loop/models/feature_binding_salience_lapse_l2.py`
**Its hypothesis:** We refine salience_lapse_endpoint_prev_l2 by incorporating perceptual feature-binding noise from feature_binding_listener_l2 into reference resolution. When resolving referring expressions in visual reference games, listeners reason under perceptual feature-binding noise (Feature Integration Theory): features active in working memory are prone to illusory conjunctions, erroneously binding to distractor objects that physically lack them. Susceptibility to illusory conjunctions scales with a distractor's visual feature complexity, directly resolving the source model's severe underprediction of literal semantic violations (false-feature choices) on complex non-matching objects while preserving bounded-capacity pragmatic inference and salience-captured attentional lapses. Differences from salience_lapse_endpoint_prev_l2: - Recursion depth: Unchanged; choice_probs calls L2. - Parameters added: w_binding (prior: dist.Beta(1.0, 24.0)). - Parameters removed: None. - Perceptual feature-binding mechanism: In choice_probs on word trials, incorporates perceptual feature binding from feature_binding_listener_l2: non-matching distractor objects receive illusory conjunction probability scaling with relative visual feature complexity and prior salience (binding_weight = is_false * (ctx.feature_count / max_fc) * prior, normalized as p_bind), mixed into the pragmatic interpretation as p_word = (1.0 - effective_bind) * heard + effective_bind * p_bind, which then undergoes salience-captured attentional lapse mixing (p_choice = (1.0 - lapse) * p_word + lapse * prior). - All other terms: Capacity-limited inferential updating (kappa) at L1 and L2, contextual feature prevalence accessibility (w_prevalence) and referential equivalence class size redundancy discounting (w_redundancy via compute_equivalence_class) on speaker alternatives in S1 and S2, literal truth-conditional semantics at L0, depth-2 RSA recursion, shared perceptual referent prior (w_singleton, w_distinct, w_features, w_empty, w_color), and unprompted prior choice remain identical to salience_lapse_endpoint_prev_l2.

**The data your statistics receive:** 7137 trials (rows) over 115
distinct displays (`objects` × `utterance`), from auto_psych. The responses CSV
(`/scratch/users/mcfrank/auto-psych/agent_trees/94d7a3a1354bcf6e/repo/_runs/outer/experiment3/model_loop/responses.csv`) holds more columns and rows; your statistic receives only
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
`/scratch/users/mcfrank/auto-psych/agent_trees/94d7a3a1354bcf6e/repo/_runs/outer/experiment3/model_loop/round_1/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior (only `choice` differs between them), and flags a statistic as a
**significant discrepancy** when its two-sided p ≤ 0.05 (with a Benjamini–Hochberg q
beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 7137 rows).
Compute display properties once per distinct display and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
