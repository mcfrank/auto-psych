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
| Designer and implementer | Gemini 4 (`gemini-4-argon`) | long-horizon tool use; smoke test §9 |
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
| Speakers | Every theory must define a speaker; the default for a theory without one is to be confirmed (§10) | the charter covers what a speaker should say |
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
| slider / allocation | points or a slider over referents | Dirichlet or beta around the posterior |
| production | pick a message for a target | speaker distribution, with a lapse |
| binary judgment | "would the speaker say X?" | Bernoulli on the speaker probability |

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
- **allocation** (response type `slider`) is a Dirichlet-multinomial around the lapse-mixed
  posterior, with concentration `kappa`; the paradigm fixes `total_points`. This is the betting
  measure of Frank & Goodman (2012); zeros are ordinary outcomes.

**The recovery test** (`recovery_check`): 3 independent datasets of 1,500 simulated responses
from a theory at known parameters are refitted. Each true value must fall inside its 98%
posterior interval, and each link parameter must be learned (posterior sd at most half the
prior's), in all replicates but at most one, and every fit must converge. One replicate is
allowed to miss because a single dataset misses a 98% interval over several parameters often
enough to fail a sound link. That happened to the allocation link's lapse on one seed, while
the likelihood's profile peaks at the truth. All three library links pass.

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

**Speakers** (PI 2026-10-21): every theory must define a speaker, because the production and
"would the speaker say X?" links need one. For a theory without one, the proposal is
`speaker(r → u) ∝ listener(r | u)`, taken over the scene's messages: the speaker who expects
this theory's own listener. It has no new parameter, because the link's temperature is its
rationality. For an RSA listener at depth n this is the speaker at depth n + 1; for a heuristic
listener it is the speaker that heuristic implies. The rule is to be confirmed (§10).

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
3. **Implementer** (Gemini 4): builds the jsPsych page from the spec, under main's validators
   (button-only responses, data contract, catch and comprehension trials).
4. **Pilot gate:** LLM participants complete it; comprehension and catch rates above a floor; the
   data parse into the declared response type; the time estimate within ±30%.
5. **Review:** the PI reviews the first paradigm of each chain; afterwards only flagged ones.
6. **Live collection**, then the inner loop.

The designer may not see participant-level data from earlier experiments, only per-paradigm
fits, so that it designs against the theories rather than individual people.

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
closed-campaign data. What differs: the designer, the implementer, and that the paradigm, and
with it the link, changes every experiment.

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
| 1 | Paradigm-spec schema and validator; link library with recovery tests; theory + link refactor; port the seeds and closed finals (exact-equivalence tests) | cloud | port and choice link done (`src/rsa_open/`); 19 of 20 ported exactly; the rest waits on §10. Paradigm spec and validator done (`src/rsa_open/paradigm.py`): budget, pay rate, N and length bounds, scene-pool bound (≤ 6 objects, ≤ 4 features, ≤ 6,000 scenes), SVG safety, content rules, link check; the closed paradigm written as a spec gives the closed design pool exactly. Fitting path for any link (`src/rsa_open/fit.py`), leaving the closed fit untouched; choice, rating and allocation links done and each passes the recovery test (3 replicates). Next: production and binary-judgment links, once §10's speaker question is settled |
| 2 | Designer tools as an agent CLI; designer brief; implementer; simulated pilot | cloud | |
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

1. **The default speaker** (§3.1): the derived speaker, `speaker(r → u) ∝ listener(r | u)`,
   for every theory without one, ported or new? Or S0/S1 from the literal semantics (the same
   for every theory)?
2. The bakeoff's hypotheses and power, at registration.
