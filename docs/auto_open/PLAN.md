# auto-psych-open: theories of pragmatic reference from open-ended paradigms

Branch `auto-psych-open`. This is the running plan and the record of decisions; update it as
phases land.

## Why

The closed RSA campaign (`docs/auto_rsa/`) found models that beat the textbook and
literature-derived models on new people: in the pre-registered test (OSF s48jv), chain 0's
final beat the best promoted seed by +197.9 ± 35.1 nats and chain 2's by +60.8 ± 22.9. But those
models are not parsimonious (chain 0's final has 17 parameters). Consolidating the three finals
gave 9 parameters at a cost of 81 nats, mostly on the displays designed to separate the finals
from the bar (`docs/auto_rsa/CONSOLIDATION.md`). Every experiment in that campaign used one
paradigm: a forced-choice reference game, with one set of materials and instructions.
Yarkoni's (2020/2022) "generalizability crisis" suggests a reading: the models may be fitted to
the paradigm, not to the phenomenon.

This campaign keeps the phenomenon and opens the paradigm. A designer agent makes a new
experiment each cycle: new materials, instructions, trial structure and dependent variable,
chosen with expected-information-gain tools. The theorists propose theories that must predict
every paradigm with one set of parameters. A final bakeoff compares this campaign's theories
with the closed campaign's on paradigms neither campaign saw.

## Decisions (PI, 2026-10-20)

| Decision | Choice | Why |
|---|---|---|
| Domain | Pragmatic reference only, bounded by a charter (§2) | keeps the closed campaign's seeds and finals comparable; the paradigm, not the phenomenon, is what varies |
| Paradigm | New every experiment: materials, instructions, trial structure, DV; chains do not share paradigms | a theory must earn its parameters across tasks |
| Theory and link | Theories are paradigm-free; a *link* maps a theory's latent quantities to each response type. Theorists never write links and never see a paradigm id | prevents "link hacking" and paradigm-specific special cases |
| Who writes links | The designer: picks one from the library or proposes one, which must pass an automatic recovery test; the link is frozen with the paradigm before any data. No human in the link loop | the designer has no stake in which theory wins, and nobody tunes a link after seeing data |
| Designer objective | Do what it judges most important for generalizable theories, given the earlier paradigms and the current theories; no paradigm-novelty rule | a novelty rule would ask for the very thing we want to observe |
| Drift | Theories are held to the old data: a candidate must not get clearly worse on any earlier paradigm (§5); the charter keeps the designer on the phenomenon. No anchor block | an anchor block would make experiments heterogeneous |
| Simulated pilot | A gate the designer must pass: launch refuses a paradigm without a passing pilot of that exact spec | catches instructions that cannot be followed, unparseable data and bad time estimates before money is spent |
| Human review | PI reviews the first paradigm of each chain (3 in all); later ones only if a validator flags them | IRB: content rules are automatic, review is a backstop |
| Content | No deception; no affectively charged stimuli; materials are SVG or text | within the existing protocol; no image copyright or Safe Browsing risk |
| Seeds | The closed campaign's seeds, *not* its finals: each chain starts from its closed counterpart's seeds (6, 5 and 5 of the 12 promoted seeds); the bar is again the 5 starting models plus the 12 promoted seeds | the bakeoff compares the two methods from the same start |
| Data | The literature data the closed campaign started from (its "E") plus this campaign's own live data. **No closed-campaign live or pre-registered data** | the methods are compared on equal data |
| Theorist | Gemini 3.8 Flash, as in the closed campaign, with the same sandbox, timeouts and round structure | only the paradigm side changes |
| Designer | Gemini 4 (`gemini-4-argon`) | long-horizon tool use; smoke test §9 |
| The page (2026-10-21, **for the PI**) | One generic, tested jsPsych page renders every paradigm from trial lists built in Python, instead of an implementer agent writing a page per paradigm | every spec uses one of six widgets with SVG or text: an agent-written page adds code to validate and nothing a spec can ask for; the pilot reads the same lists the page shows. Reversible |
| Final bakeoff | Two arms (forced choice, which favours the closed theories; a blind new paradigm, which favours generalization); fit A primary (both sides on the same pooled data), fit B secondary (each on its own campaign's data) (§7) | |
| Blind paradigm | Designed by an isolated designer run that no chain sees | |

## Decisions (PI, 2026-10-21)

| Decision | Choice | Why |
|---|---|---|
| Porting rule | As in §3, approved; its exact mechanics and the cases it does not settle are in §3.1 | |
| Budget parity | **Equal dollars** per experiment: the closed campaign's ~$133 (200 × $0.50 × Prolific's 1.33), about 600 participant-minutes at the same hourly rate. The designer chooses N and length within it | the comparison is of methods at equal cost |
| Old-data guard | k = **4** clustered SEs, as the closed campaign's guard | |
| Pilot participants | Gemini 3.8 Flash | cost |
| Response process | Every link has a decision temperature and a lapse; a temperature written into a ported theory is cut (one hand port) | standard in a link; equal flexibility for every theory, old ones included |
| Chain 2's final | Ported whole (rule b); its prior-mixing lapse stays in the theory as `theory_lapse`, and the link adds its uniform lapse | the theory's mixture must earn its parameter |
| Sliders and bets (PI 2026-10-21) | A one-referent slider or bet uses the ordered-beta link (a discrete Beta between point masses at 0 and max); a bet over all referents (F&G 2012) keeps the Dirichlet-multinomial, response type `bet` | sliders and bets are prominent in the RSA literature; endpoints pile up in real slider data |
| Timing digest (PI 2026-10-21) | The designer's brief carries how long finished experiments took people (aggregates only), the closed campaign's 9 live experiments and every chain's of this campaign; the pilot's time estimate is calibrated on the same records | helps the designer judge how many trials fit in its minutes; durations are not response data, so this is not closed-campaign data in the sense of the drift rule |
| Sliders: click-to-place (PI 2026-10-21) | No handle until the first click, which places it; a one-line hint; Continue until placed | a visible default draws answers to itself on 0-100 sliders (Liu & Conrad, 2019); a pile-up at the midpoint would read to the ordered-beta link as real interior mass |
| Materials legibility (PI 2026-10-21) | The brief asks for recognisable bases, nameable features and engaging drawings; the pilot gates it: fresh chats name each base and each single added feature, ≥ 90% right and no item misread by half | the first designer drew a bare box as a featureless robot |
| Speakers | Every theory has a speaker: its own `speaker(params, ctx)`, or the one derived from its listener, `speaker(u | r) ∝ listener(r | u)` over the scene's real messages (§3.1) | the charter covers what a speaker should say; a derived speaker is specific to each theory, where S0/S1 would give every listener-only theory the same one |
| Designer limits | Per cycle: 2 h of wall clock, 40 tool calls, at most 3 simulated pilots. `submit` refuses a paradigm whose EIG per dollar is below half the closed campaign's paradigm's at the same budget, under the same theories. A failed cycle gets **one retry** with the reason in the brief; a second failure stops the chain and reports. Design bounds: N ≥ 100, at most 15 minutes per person | bounded cost; informative without dictating the design; fail loudly, no fallback design |

## 1. Claims

1. **Transfer.** On the blind paradigm, this campaign's final theories predict new people better
   than the closed campaign's finals (and than the seeds).
2. **Parsimony.** At equal or better transfer, they use fewer theory parameters, and their
   parameters are more stable across paradigms (each paradigm fitted alone, estimates compared).
3. **The designer.** Agent-designed paradigms are valid and informative: realized information
   gain against predicted EIG, comprehension and catch pass rates, pilot against live.

## 2. The charter

Anchored in Grice (1975) and Goodman & Frank (2016, *Trends in Cognitive Sciences*, "Pragmatic
language interpretation as probabilistic inference").

- **In scope:** how people use a message, its alternatives and the shared context to infer what
  a speaker means (a referent, an implied meaning) or to choose what a speaker should say: tasks
  in which a message is interpreted or chosen given a context both parties know.
- **Out of scope:** perception or memory limits as the object of study; learning over many
  trials; social evaluation of speakers; anything needing deception or emotionally charged
  material.
- **Every paradigm maps each trial onto the abstract scene:** the referents and their features,
  the message (or its absence), the alternatives the speaker had, and what the participant is
  asked to report.

The designer's brief quotes the charter, and a validator checks each paradigm spec against it.

## 3. Theories, links and paradigms

**A theory** (written by the theorists) is a memo file, as now, but paradigm-free. Given an
abstract scene it returns latent quantities: the listener's posterior over referents, and the
speaker's distribution over messages. It has one parameter set for every paradigm and cannot
see which paradigm a trial came from.

**A link** (from the library or the designer) maps those latents to the likelihood of one
response type, with a few nuisance parameters of its own, shared by every theory on that
paradigm. The first library, written and tested before the campaign:

| link | response | maps |
|---|---|---|
| choice | pick one referent | listener posterior, with a lapse to uniform |
| rating | Likert-type rating of one referent | ordered probit on the posterior probability |
| slider (`ordered_beta`) | a 0–max slider or bet on one referent | ordered beta (Kubinec, 2022): point masses at 0 and max, a discrete Beta between, around its tempered probability |
| bet (`allocation`) | split points over the referents (Frank & Goodman, 2012) | Dirichlet-multinomial around the tempered posterior |
| production | pick a message for a target | speaker distribution, with a lapse |
| binary judgment | "would the speaker say X?" | probit on the logit of the speaker probability, with a bias |

Each library link has a recovery test (simulate from known parameters, refit, recover), and the
choice link must reproduce today's forced-choice likelihood exactly. A designer-proposed link
must pass the same recovery test automatically before it can be used.

**Built so far** (`src/rsa_open/links.py`, `src/rsa_open/fit.py`). The closed fit (`src.rsa.fit`)
knows only a choice likelihood and is left untouched, so the closed campaign's results cannot
move. The open fit takes the likelihood from the link:
- **choice** gives the closed fit's per-trial log-likelihood exactly (the same data and
  parameters).
- **rating** is an ordered probit on `beta · logit(p)` of the probed referent, with equally
  spaced cutpoints (`center`, `spread`) and a uniform-rating lapse; the paradigm fixes
  `scale_points`.
- **allocation** (response type `bet`) is a Dirichlet-multinomial around the lapse-mixed
  posterior, with concentration `kappa`; the paradigm fixes `total_points`. This is the betting
  measure of Frank & Goodman (2012); zeros are ordinary outcomes.
- **ordered_beta** (response type `slider`) is a 0–max slider on one probed referent: point
  masses at 0 and max from two cutpoints on the logit of its tempered probability, a discrete
  Beta (precision `phi`) between them (Kubinec, 2022).
- **production** and **binary_judgment** score the theory's speaker (§3.1).

**The recovery test** (`recovery_check`): 3 independent datasets of 1,500 simulated responses
from a theory at known parameters are refitted. Each true value must fall inside its 98%
posterior interval, and each link parameter must be learned (posterior sd at most half the
prior's), in all replicates but at most one, and every fit must converge. One replicate is
allowed to miss because a single dataset misses a 98% interval over several parameters often
enough to fail a sound link. That happened to the allocation link's lapse on one seed, while
the likelihood's profile peaks at the truth. All six library links pass.

**A paradigm spec** (written by the designer) is a structured file: charter fields; materials
(SVG or text); instructions; trial structure; response type and link; the parameterized trial
space; the mapping from each trial to an abstract scene; time estimate and pay. A schema
validator checks it, and its sha256 is recorded with every trial collected under it.
`src/rsa_open/paradigm.py` is the format and its validator. The scene space is a set of
display sizes (objects × features, as in the closed campaign's pool); every trial is a
`src.rsa.context.Context`. The validator reports every problem at once, so the designer can fix
them in one pass.

**Data** generalize the RSA trial schema: `paradigm_id`, the abstract scene as JSON, the response
as JSON, participant and trial ids. Theories see only the scene.

**Porting the seeds and the closed finals** to theory + link is mechanical, by a rule fixed before
the campaign: response-process parts (lapse, a decision temperature applied to the listener
posterior) go into the link; anything that changes the latent posterior or speaker stays in the
theory. Each port must reproduce its original forced-choice predictions exactly on the old
data. The closed finals are ported, tested and frozen before the open campaign starts, so
nothing in it can influence them.

### 3.1 The port, exactly (`src/rsa_open/port.py`)

**Every link has a decision temperature and a lapse** (PI 2026-10-21; `src/rsa_open/link_kit.py`):
the theory's posterior is tempered (p ** temperature, renormalized; prior LogNormal(0, 0.5),
median 1), then mixed with a uniform guess with probability lapse (prior Beta(1, 9)), then
mapped to the response. Every ported model gains the temperature parameter. That adds
flexibility to the old models, and it is the standard response process; it also means no
theory gains by writing its own.

A closed model file defines `PARAMS` and `choice_probs(params, ctx)`. Two mechanical rules:
- **(a) a final uniform lapse.** Every `return` of `choice_probs` is
  `with_lapse(X, params["lapse"])` and `PARAMS["lapse"]` is the link's `Beta(1, 9)`: `lapse`
  leaves `PARAMS` and each return becomes `return X`. Composed with the choice link, the port
  equals the original at temperature 1.
- **(b) no final lapse.** No return is `with_lapse(...)`: the whole function is the theory, and
  a parameter named `lapse` (a latent mixture of the theory's own) is renamed `theory_lapse`.
  Composed with the choice link, the port equals the original at link lapse 0 and temperature
  1. The theory's own mixture stays only if it earns its parameter over the link's uniform
  lapse (PI 2026-10-21).

Either way the function is renamed `listener`, and only those spans of the source text are
edited: every memo line stays byte for byte (memo parses its own source). A file that fits
neither rule is refused with the reason. There is no fallback.

**Hand ports.** A decision temperature written into a theory belongs to the link, but no
mechanical rule can find one in arbitrary code. Of the 20 models, one has one, and it is
ported by hand (`src/rsa_open/hand_ports/`, registered in `HAND_PORTS` with the original's
sha256, so a changed original is refused).

**Composing** a theory with a link gives an ordinary model file (`compose`): the theory, then
the link's parameters and its `choice_probs`. The existing fitting, CV and loop code runs it
unchanged.

**The 20 models** (5 starting models, 12 promoted seeds, 3 closed finals; `tests/test_rsa_open_port.py`):
- 18 port by rule (a);
- chain 2's final (`competition_endpoint_binding_lapse_l2`) ports by rule (b). Its lapse
  mixes in its own salience prior, on word trials only;
- `softmax_surprisal_isolated_costly_l4` (promoted) is the hand port. Its temperature `beta`
  (heard ** beta, renormalized) is cut. It equals its original on word trials at temperature
  = beta, and on every trial at beta = temperature = 1. On "mumble" trials the original left
  the prior untempered, and the link tempers every trial.

Each port equals its original at those values, and every port gains the link's temperature.

**Speakers** (PI 2026-10-21; `src.rsa_open.fit.TheoryModel`). Every theory has a speaker,
because the production and "would the speaker say X?" links need one: its own
`speaker(params, ctx)` (an N_UTT × N_OBJ array, column r the distribution over messages for
target r), or the speaker derived from its listener:
`speaker(u | r) ∝ listener(r | u)` over the scene's real messages, never the sink. That is the
speaker who expects this theory's own listener. It has no parameter of its own, because a
speaker link's temperature is its rationality. For an RSA listener at depth n it is the speaker
at depth n + 1; for a heuristic listener, the speaker the heuristic implies. When no real
message is true of the target for this listener, the speaker is indifferent: uniform over the
real messages. Every ported theory uses the derived speaker; the open campaign's theorists may
write their own.

## 4. The designer

**Each experiment:**
1. **Brief:** the charter; every earlier paradigm in the chain, with each live theory's fit on
   it; the current theories and their disagreements; the link library; the budget; the goal:
   design the experiment most important for arriving at generalizable theories.
2. **Tools** (a sandboxed CLI, like `check_candidate`; the harness computes every score):
   - `eig` scores a paradigm's trial space under the live theories and the bar (with quotas
     inside, as now), and reports EIG per participant-minute;
   - `power` simulates the experiment;
   - `link_check` runs a proposed link's recovery test;
   - `pilot` builds the page and runs LLM participants through it;
   - `submit` freezes the spec and its link and refuses without a passing pilot of that spec.
3. **The page** (`src/rsa_open/screens.py`, `page.html`; replaces the planned implementer, a
   decision for the PI): trial lists are built in Python from the frozen spec and design, and
   one generic jsPsych page plays them (§4.2).
4. **Pilot gate:** LLM participants complete it; comprehension and catch rates above a floor; the
   data parse into the declared response type; the time estimate within ±30%.
5. **Review:** the PI reviews the first paradigm of each chain; afterwards only flagged ones.
6. **Live collection**, then the inner loop.

The designer may not see participant-level data from earlier experiments, only per-paradigm
fits, so that it designs against the theories rather than individual people.

### 4.1 The EIG tool (`src/rsa_open/design.py`, built 2026-10-21)

- **The design context** is written by the harness and only read by the designer: the live
  theories, each one's posterior draws on the data so far, and a uniform prior over them.
- **Trial units** are what a design picks:
  - a scene (choice);
  - a scene and a probed object (rating);
  - a scene and a target (production);
  - a scene, a target and a message (judgment).

  A probe or target is taken once per class of identical objects.
- **Each unit gets** `n_participants × trials_per_participant / distinct_trials` responses. The
  paradigm spec now gives `distinct_trials` (the closed campaign's D = 52).
- **The estimator** is the closed campaign's (`src/rsa/design/eig.py`): greedy joint-EIG
  selection, then the joint EIG and power on fresh scenarios.
- **The floor** is half the bits per dollar of the closed paradigm (1,597 scenes, 200 × 13 of
  52, the choice link) under the same theories.
- **Sliders** (the ordered-beta link) are scored on their responses binned into 12 outcomes
  (0, ten interior bins, max). Coarsening can only lose information, so the score is a lower
  bound. **Bets** (allocation) are refused until there is an estimator for them.

**Link parameters (approved by the PI, 2026-10-21).** A new paradigm's link has no data, so its parameters
come from the link's prior. Averaging them inside each theory's evidence (the estimator as
written) is a draw lottery: with diffuse nuisance parameters and ~1,000 responses, a theory's
evidence is whichever single draw happens to fit. The measurements, on a rating paradigm with
two theories:

| draws | bits | correct |
|---|---|---|
| 32 | 0.96 | 72% |
| 256 | 0.98 | 91% |
| link fixed | 1.00 | 100% |

Links with more nuisance parameters would be scored worse, arbitrarily.

So the tool scores the information about the theories **given** the link's parameters:
- 8 settings are drawn from the link's prior;
- every theory's posterior draws are paired with each;
- a simulated scenario compares the theories at its own setting (`draw_groups`, a
  default-preserving option in the closed estimator; tests in `tests/test_rsa_design_eig.py`).

On the same rating paradigm this gives 0.99 bits and 99.8% correct. The four categorical links
score 0.90–0.99 bits with 97–100% correct.

**The cost:** a theory difference the link can mimic (a sharper listener against a higher
temperature) counts as information, though the joint fit may not separate it.

### 4.2 The tools, the page and the pilot (built 2026-10-21)

**The tools** (`src/rsa_open/tools.py`) run in the designer's sandbox, on its workspace
(`context/` and `inputs/` read-only, `work/` its own):
- `validate`: every problem with a spec, at once;
- `eig`: §4.1, with a seed fixed by the spec's sha256, against a floor scored once per context;
- `link-check`: the recovery test in this paradigm (its scene space, number of responses and
  scale), simulated from a live theory at its posterior mean (2–20 minutes);
- `submit`: records the spec once it has a floor-meeting `eig` and a recovering `link-check` of
  that exact spec.

Every call is logged and the 41st refused.

**Prompts carry the message.** A listener or judgment prompt shows the speaker's word where it
says `{message}`; listener paradigms with prior queries give a `prior_prompt`; ratings and
sliders name their ends (`scale_labels`).

**The page** (`src/rsa_open/screens.py`, `page.html`, `page.py`). Trial lists are a pure
function of (spec, design, seed, list):
- each list takes `trials_per_participant` units of one shuffled order of the design, so every
  unit is answered equally often to within one;
- catch trials, with pass rules, go at random positions, after a practice trial;
- domains rotate, features map to words at random, and object order is random;
- objects are drawn (SVG) or written (text).

One jsPsych plugin handles all six response types (a slider must be touched; a bet's points
must add up). A wrong comprehension answer shows the instructions once more. Responses are
recorded in the link's form, as scene indices.

**The pilot** (`src/rsa_open/pilot.py`). Gemini 3.8 Flash participants take the same lists, one
chat each; an SVG trial is one image (resvg), the marked object framed. The gate requires:

| check | threshold |
|---|---|
| comprehension, by the second reading | ≥ 80% |
| catch trials passed | ≥ 80% |
| usable answers | ≥ 95% |
| test data | parse as the link's data |
| time estimate (`src/rsa_open/timing.py`: a reading-and-answering model, scaled by people's median times in finished experiments of the same response type, else of every type) | within 30% of the declared minutes |
| test answers | not all identical |
| legibility (SVG): each base named among five kinds, each single feature named among the domain's words, by 3 fresh chats | ≥ 90% right, no item misread by half |

A first live run passed: 3 participants, choice, SVG creatures, every check and catch trial right.

**Timing** (`src/rsa_open/timing.py`). The uncalibrated model predicted 4.9 minutes for the closed
paradigm, where people's median was 2.1 (9 experiments, about 1,800 people), so the estimate is
scaled by people's real times: by finished experiments of the same response type when there
are any, else by all. Until this campaign has run a type, it is calibrated on the closed
campaign's forced choices. `validate` and `eig` report the estimate against the declared
minutes, so the designer sees the pilot's time check before submitting. The brief's
`inputs/timing.md` lists every record.

### 4.3 The designer cycle (`src/rsa_open/designer.py`)

**Each cycle:**
1. The harness lays out a workspace: the context, with the floor scored; the schema; a
   placeholder format example; the link library; the chain's earlier paradigms and per-theory
   fits.
2. It spawns Gemini 4 with the brief: the goal (theories that generalize across paradigms), the
   charter, the theories, the bounds, the tools, and a `rationale.md` to write.
3. After each session the harness checks the submission itself:
   - the spec validates and is the one submitted;
   - its design is re-scored with the harness's seed and must meet the floor; that design runs;
   - its link is re-checked;
   - its lists are piloted.
4. A failed submission goes back with the report. A cycle allows up to 3 submissions, and an
   unchanged failed spec is refused.

**Limits:** 2 h of agent time, 40 tool calls. A session without a submission ends the cycle. A
failed cycle gets one retry with its reason; then `DesignerFailed` stops the chain.

**The first real cycles (2026-10-21, smoke, 20-minute sessions).** Gemini 4 scored forced-choice
and production paradigms (all above the floor), switched to production when its choice
paradigm's link check failed, submitted, and passed the harness's re-score, link re-check and the
pilot's comprehension, catch, readability and parse checks. It failed only the time check
(estimated 12.0 minutes against 4.4 declared), and the smoke's short sessions ended the cycle.
What it found:

| problem | fix |
|---|---|
| the link check simulated from the first theory by name, the literal listener, under which a listener link's temperature changes nothing: every choice paradigm failed | the theory with the most parameters, unless named (`--theory`) |
| no tool showed the time estimate before the pilot | `validate` and `eig` report it |
| the estimate was an uncalibrated guess, about twice too slow | calibrated on completion times (above) |
| a session cut off by its time limit was reported as "ended without a submission" | reported as reaching its limit |
| a prompt with quotation marks around `{message}` shows the word twice-quoted | the validator refuses it; the brief says so |
| opencode without its provider's key read as two empty sessions | a failed login, raised (`src/runtime/usage_limits.py`) |

**Freezing** writes the spec, design, lists, a preview page, the link check, the pilot and
`record.json`. The record holds the hashes, `review_required` for a chain's first experiment,
and the code commit.

## 5. The inner loop

As in the closed campaign (admission gates, grouped CV, the ledger, the critic, the 2-stale-round
stop, carry-forward of at most 8 distinct theories), with three changes:

- candidates are theories only; every fit is theory × the paradigm's frozen link;
- grouped CV folds are built within each paradigm;
- **accountability to old data:** a candidate replaces the incumbent only if it is better on the
  new paradigm *and* not more than k clustered SEs worse on each earlier paradigm, the
  literature data E included.

## 6. Parity with the closed campaign

The same seeds and chain split, the same literature data, the same theorist model and sandbox,
the same rounds and slots, 3 chains × 3 experiments, the same per-experiment budget, and no
closed-campaign data. What differs: the designer, the page built from its spec, and that the
paradigm, and with it the link, changes every experiment.

## 7. The final bakeoff (pre-registered, as `rsa-prereg-final` was)

**Competitors:** the three open finals and the three closed finals (ported, §3), plus the seeds.

**Arms:**
- **F, forced choice:** the closed campaign's paradigm on fresh displays, designed with its own
  rule. This is the closed theories' home ground.
- **T, transfer:** a blind paradigm with a new DV, designed by an isolated designer run that no
  chain or theorist has seen.

**Fits:**
- **A (primary):** every competitor fitted on the same pooled data (E plus both campaigns' live
  data; never the closed campaign's pre-registered data), so the comparison is of theories.
- **B (secondary):** each fitted on its own campaign's data, so the comparison is of methods,
  including the data each chose to collect.

**Predictions** (to be fixed at registration): the open finals beat the closed finals in T; F is
tested the same way, where the closed finals are expected to do at least as well.

## 8. Content and IRB

Validators refuse any paradigm with deception, affectively charged stimuli, or materials other
than SVG or text; plain-language instructions; pay at the platform rate for the piloted time.
The existing protocol covers the rest (PI, 2026-10-20).

## 9. Phases

| Phase | What | Where | Status |
|---|---|---|---|
| 0 | This plan; Gemini 4 smoke test | cloud | plan written; smoke test below |
| 1 | Paradigm-spec schema and validator; link library with recovery tests; theory + link refactor; port the seeds and closed finals (exact-equivalence tests) | cloud | **done** (`src/rsa_open/`): all 20 models port (18 by rule a, chain 2's final by rule b, one by hand) and equal their originals; five links (choice, rating, allocation, production, binary judgment), each with a temperature and a lapse, each passing the recovery test; every theory has a speaker; the paradigm spec and its validator. Not yet: freezing the ported closed finals (with their derived speakers) before the campaign |
| 2 | Designer tools as an agent CLI; designer brief; the page; simulated pilot | cloud | **built** (§4.1–4.3): the EIG tool; the tools CLI (`validate`, `eig`, `link-check`, `submit`, 40 calls); spec-driven trial lists and one page for all six response types (tested in Chromium); the pilot with Gemini 3.8 Flash participants and its gate (a live 3-participant run passed); the designer cycle (brief, the harness's own re-checks, up to 3 pilots, one retry, freeze). Not yet: a real Gemini 4 designer cycle end to end; EIG for bets; designer-proposed links |
| 2b | Fitting across paradigms (shared theory parameters, each paradigm its own link parameters, E included); the open inner loop (theorists; grouped CV within paradigm; the k = 4 SE accountability rule; carry-forward); the chain driver (context → designer → frozen paradigm → simulated collection → inner loop → timing record); rehearsal ground truths; Sherlock jobs | cloud | **the gate for Phase 3**, with a designer cycle that freezes a paradigm and a local one-experiment dry run |
| 3 | Rehearsal on simulated participants with a ground-truth theory across paradigms | Sherlock | |
| 4 | One live paradigm per chain, with PI review | live | |
| 5 | Campaign: 3 chains × 3 experiments | live | |
| 6 | Bakeoff: arms F and T, pre-registered | live | |

**Gemini 4 smoke test (2026-10-20).** The API key sees `gemini-4-argon`. A designer-style call
(propose one new paradigm, JSON only) returned valid JSON in 6 s for about 430 tokens. It
proposed a betting measure, the DV of Frank & Goodman (2012). Through opencode 1.18.35 (the agent
runner): its catalog does not list the model yet, so it must be declared in `opencode.json`
(`provider.google.models["gemini-4-argon"]`); declared, it answered and wrote a file as an agent.
Gemini 3.8 Flash, the theorist, runs as before.

## 10. Open

1. **Designing trials beyond the scene.** A rating's probed object, and a speaker trial's target
   and judged message, are design choices too: the EIG tools must score (scene, probe or target,
   message) units, not scenes alone (Phase 2).
2. The bakeoff's hypotheses and power, at registration.
