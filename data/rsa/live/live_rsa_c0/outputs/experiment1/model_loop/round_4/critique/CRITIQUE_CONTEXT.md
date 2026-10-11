# Critique context

**Incumbent (best) model:** `expected_success_confusion_attention`
**Its code:** `/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment1/model_loop/models/expected_success_confusion_attention.py`
**Its hypothesis:** Pragmatic listener at depth 2 with bounded attentional capture, expected communicative success, competitor confusion avoidance, and visual salience prior. Refining expected_success_confusion_speaker by incorporating the capacity-limited attentional capture component from limited_attention_listener: listeners interpret referring expressions under capacity-limited visual attention, balancing top-down pragmatic reasoning against bottom-up visual attentional capture. Under higher visual load in larger displays, attentional dispersion occasionally causes bottom-up attentional capture by salient objects without full communicative verification, whereas unambiguous pop-out displays ensure near-perfect target identification; in the attentive mode, listeners invert a depth-2 speaker maximizing linear expected communicative success while penalizing competitor confusion. This resolves excessive scalar exclusion and elevates choices of feature-rich objects on ubiquitous and scalar displays while naturally capturing foil selections under visual load.

Differences from expected_success_confusion_speaker:
- Recursion depth remains depth 2 (choice_probs calls L_attention, which inverts a speaker simulating depth-1 listener L1).
- Added parameters: logit_p_base (prior dist.Normal(2.0, 1.0)), w_load (prior dist.Normal(0.5, 0.5)), and w_popout (prior dist.Normal(1.0, 0.5)).
- No parameters removed (retains alpha, w_singleton, w_distinct, w_features, w_valence, w_color, w_confusion, and lapse).
- Replaced depth-2 listener L2 with L_attention, adding global domain MODES = jnp.arange(2), an attentional mode choice (listener: chooses(mode in MODES, wpp=p_attend if mode == 1 else (1.0 - p_attend))), and referent choice conditioned on mode (listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] if mode == 1 else vec(prior, r))).
- In choice_probs, computed is_popout and display-size-modulated attention probability p_attend = jax.nn.sigmoid(logit_p_base + w_popout * is_popout - w_load * (n_obj - 2.0)), passing p_attend to L_attention.

**The data your statistics receive:** 2775 trials (rows) over 193
distinct displays (`objects` × `utterance`), from auto_psych. The responses CSV
(`/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment1/model_loop/responses.csv`) holds more columns and rows; your statistic receives only
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
`/scratch/users/mcfrank/auto-psych/agent_trees/63cfc8f3a326e982/repo/_runs/outer/experiment1/model_loop/round_4/critique/test_stats/<name>.py` as `test_statistic(df)` returning one
float, with `# name:` and `# description:` header comments.

You do not need to run anything: the pipeline computes each statistic on the
observed data and on 1000 datasets simulated from the incumbent's
posterior (only `choice` differs between them), and flags a statistic as a
**significant discrepancy** when its two-sided p ≤ 0.05 (with a Benjamini–Hochberg q
beside it).

**Time:** each statistic is called 1001 times. Each call must finish
within 5 s and all of them within 300 s
(about 0.3 s per call on 2775 rows).
Compute display properties once per distinct display and map them onto the rows; a
statistic that is too slow, raises or returns a non-finite value is set aside and
reported back to you.
