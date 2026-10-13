# Critique context

**Incumbent (best) model:** `prototype_typicality_bounded_listener`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment2/model_loop/models/prototype_typicality_bounded_listener.py`
**Its hypothesis:** Pragmatic listener at depth 2 under capacity-limited executive inhibitory control with contextual category typicality and bounded speaker consideration. Refining prototype_typicality_listener by incorporating bounded speaker consideration from bounded_consideration_speaker: speakers evaluate communicative alternatives probabilistically rather than exhaustively, retrieving distinguishing alternative expressions with a bounded consideration rate and otherwise producing an unconsidered true feature without communicative comparison. Pragmatic listeners invert this bounded-consideration speaker under capacity-limited executive inhibitory control, recognizing that hearing an ambiguous word does not rule out feature-rich competitors whose unique distinguishing words failed to enter the speaker's consideration set. This directly moderates overly aggressive scalar exclusion, elevates choices of maximal-feature candidates under ubiquitous and multi-match expressions (all_match_max_feature_choice_rate and four_object_three_candidate_max_feature_rate), elevates selections of duplicated candidates under shared words (duplicate_matching_candidate_choice_rate), and calibrates pragmatic target rates against all-feature competitors (four_object_target_vs_all_feature_rate).

Differences from prototype_typicality_listener:
- Recursion depth remains depth 2 (choice_probs calls L_typical, which inverts a speaker simulating depth-1 listener L1).
- Added parameter: logit_p_consider (prior dist.Normal(1.0, 1.0)).
- No parameters removed (retains alpha, logit_p_base, w_load, w_popout, w_candidate, w_inhibit_leak, w_typical, w_singleton, w_distinct, w_features, w_valence, w_color, w_confusion, and lapse).
- Speaker utility in both S1 and S2: added an alternative consideration mode choice (speaker: chooses(mode in MODES, wpp=p_consider if mode == 1 else (1.0 - p_consider))) where communicative utility exp(alpha * L - w_confusion * at(confusion, u, r)) is evaluated when mode == 1, while mode == 0 selects true utterances uniformly with weight 1.0.
- In choice_probs: computed p_consider = jax.nn.sigmoid(params["logit_p_consider"]) and passed p_consider to L_typical.

**The data your statistics receive:** 4914 trials (rows) over 85
distinct displays (`objects` × `utterance`), from auto_psych. The responses CSV
(`/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment2/model_loop/responses.csv`) holds more columns and rows; your statistic receives only
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
`/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment2/model_loop/round_4/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior (only `choice` differs between them), and flags a statistic as a
**significant discrepancy** when its two-sided p ≤ 0.05 (with a Benjamini–Hochberg q
beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 4914 rows).
Compute display properties once per distinct display and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
