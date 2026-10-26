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
| Data | The literature data the closed campaign started from (its "E") plus this campaign's own live data. **No closed-campaign live or pre-registered data**, nor anything chosen or built on them: E is run 2's `all_trials.csv` (the five literature sources, built before any live experiment); the agents' trees exclude the closed finals, the code that consolidates them (their mechanisms), the pre-registration and the campaign figures (`agent_tree.exclude`; a test checks a built tree names no final). Kept, as ruled: the closed live experiments' completion times (the timing digest), and the closed paradigm as the floor's reference design | the methods are compared on equal data: an open theorist who read the closed finals would carry the closed experiments into the open campaign (PI 2026-10-21) |
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
| Response process | Every link has a decision temperature and a lapse; ~~a temperature written into a ported theory is cut (one hand port)~~ superseded 2026-10-22 (below): E's temperature is fixed at 1 and a theory's own temperature stays | standard in a link; equal flexibility for every theory, old ones included |
| Chain 2's final | Ported whole (rule b); its prior-mixing lapse stays in the theory as `theory_lapse`, and the link adds its uniform lapse | the theory's mixture must earn its parameter |
| Sliders and bets (PI 2026-10-21) | A one-referent slider or bet uses the ordered-beta link (a discrete Beta between point masses at 0 and max); a bet over all referents (F&G 2012) keeps the Dirichlet-multinomial, response type `bet` | sliders and bets are prominent in the RSA literature; endpoints pile up in real slider data |
| Timing digest (PI 2026-10-21) | The designer's brief carries how long finished experiments took people (aggregates only), the closed campaign's 9 live experiments and every chain's of this campaign; the pilot's time estimate is calibrated on the same records | helps the designer judge how many trials fit in its minutes; durations are not response data, so this is not closed-campaign data in the sense of the drift rule |
| Sliders: click-to-place (PI 2026-10-21) | No handle until the first click, which places it; a one-line hint; Continue until placed | a visible default draws answers to itself on 0-100 sliders (Liu & Conrad, 2019); a pile-up at the midpoint would read to the ordered-beta link as real interior mass |
| Materials legibility (PI 2026-10-21) | The brief asks for recognisable bases, nameable features and engaging drawings; the pilot gates it: fresh chats name each base freely, name each single added feature, and say which objects of real trial screens a word is true of; ≥ 90% right and no feature or trial screen misread by half; a base named otherwise by half counts against the 90% but alone is only a warning (PI, same day) | the first designer drew a bare box as a featureless robot, and a multiple-choice naming question (the domain among four unrelated kinds) passed it by elimination; then real Gemini named a plain badge "circle" and a plain robot "square" (0 of 3 each, 90.5% overall), a fair reading of an abstract base that the features and the words' truth still have to carry |
| Speakers | Every theory has a speaker: its own `speaker(params, ctx)`, or the one derived from its listener, `speaker(u | r) ∝ listener(r | u)` over the scene's real messages (§3.1) | the charter covers what a speaker should say; a derived speaker is specific to each theory, where S0/S1 would give every listener-only theory the same one |
| Designer limits | Per cycle: 2 h of wall clock, 40 tool calls, at most 3 simulated pilots. `submit` refuses a paradigm whose EIG per dollar is below half the closed campaign's paradigm's at the same budget, under the same theories. A failed cycle gets **one retry** with the reason in the brief; a second failure stops the chain and reports. Design bounds: N ≥ 100, at most 15 minutes per person | bounded cost; informative without dictating the design; fail loudly, no fallback design |

## Decisions (PI, 2026-10-22)

| Decision | Choice | Why |
|---|---|---|
| E is the reference paradigm | The literature block's link temperature is fixed at 1; its lapse, and every new paradigm's temperature and lapse, are fitted (`corpus.LITERATURE_FIXED`). A theory in which people sharpen or soften their beliefs into a choice says so with a parameter of its own, shared by every paradigm; the theorists' prompt says so. The one hand port is undone (`beta` stays in its theory) | On E alone a free temperature traded off against a theory's own sharpness: on pragmods, `rsa_l1_salience` had `alpha` 0.15 ± 0.065 at temperature 2.5 free, 1.47 ± 0.10 at 1 fixed, and `rsa_l1_shared_prior`'s two-chain fit stalled for over an hour free, against 77 s fixed (run A's design waited 6 h+ on it). That cost fell on theories with their own sharpness parameter, most RSA theories, and not on heuristics. With E fixed, every open theory is its closed original on E, and each new paradigm's temperature is its noise relative to E. The worry (PI): a theory that rightly re-attributes `alpha` to the listener's choice is squashed; the answer is that the claim moves into the theory, where every paradigm tests it, and **the audit** (`temperature_audit.py`, `open_temperature_audit.sbatch`) measures what fixing costs each bar theory on the full E. If many fit E clearly better free, a tight prior around 1 replaces the fixed value |
| The audit's verdict (2026-10-22) | **Keep E's temperature fixed at 1.** On all of E (50,587 trials; `data/rsa_open/temperature_audit/audit.md`), 12 of the 15 bar theories finished both ways. Seven lose nothing (free temperature 0.90–1.14, within ±1 nat). Five gain 12–73 nats with a free temperature of 1.2–1.9; only `evaluative_prominence_l2` clears 2 clustered SEs (+71.6 ± 28.9), and `rsa_l1` (+49.9 ± 28.5) and `rsa_l1_salience` (+73.3 ± 43.9) are near it. The free fits are unstable: two timed out at 6 h, one did not converge; every fixed fit finished with 0 divergences | the rule set beforehand was "a tight prior if many fit clearly better free"; one is clearly better, two are close, so not many. The gain those theories show is a listener sharpening their alpha cannot mimic, which the theorists may now add as a theory parameter, tested across paradigms |
| Faster loop fitting (PI, same day, from the profile of run A) | A cross-validation fold starts at its theory's full fit (posterior medians) and runs 300 warm-up + 500 draws per chain instead of 1,000 + 1,000; the job takes the node's 24 cores; the theorists' self-check fits a fixed fifth of E's displays. Full fits, admission fits and the 30-min admission limit are unchanged | the folds were ~85% of the loop's fitting (run B was estimated at 2–3 days). Checked on pragmods (3 folds): held-out lpd −5,081.5 warm against −5,083.8 cold (`rsa_l1_salience`), −5,005.9 against −5,006.1 (`rsa_l2`), all converged. The closed campaign ran cold folds at 1,000 + 1,000: a difference in compute, not in what is estimated |
| Design fits | A design theory takes the previous loop's fit on the same data, else a cheaper fit (2 chains × 500 + 250) within 2 h; a fit that fails or runs out is left out of that design, recorded in `design_screened_out.json` | the same stalled fit; no single fit may hold a chain without limit, and no theory leaves a design silently |

## Decisions (PI, 2026-10-23)

| Decision | Choice | Why |
|---|---|---|
| Run B's slowness is an inference problem, and is treated as one | Every fit records its leapfrog steps per draw, step size and timing (`src/rsa_open/sampler.py`). The self-check prints them, every parameter pair correlated above 0.85 (rank correlation), and the admission fit's projected time on all of E: the short fit's leapfrog steps, taken to the full fit's iterations, times one gradient of all of E timed for 4 chains. On pragmods it came within 40% of two full fits (`rsa_l1_salience` 114–170 s vs 168 s, `isolated_graded_costly_l3` 846–1,005 s vs 867 s), where scaling the short fit's own time was 2–3× long. A projection over 30 min is a FAIL. A non-convergence rejection carries the same report; a too-slow one carries its warmup's report when warmup finished | run B rejected 16 of 27 candidates as too slow (10 of 11 in round 1), and a theorist could not see why from a timeout. Locally on pragmods, cost per step is the same across theories (16–21 ms); what differs is steps per draw. `isolated_graded_costly_l3` takes 12.9 steps per draw at step size 0.29, its `alpha` and cost weight 0.92 correlated, against `rsa_l1_salience`'s ~7. A ridge, not a slow theory |
| No invitation to a listener-sharpness parameter; RSA's identification pairs in the prompt | The prompt no longer suggests a `beta` on the listener. It names rationality vs cost, rationality vs listener sharpness, and prior weights vs lapse. A theory keeps both members of a pair only with a stated reason in the data and a self-check that does not flag them. **This supersedes 2026-10-22's "a theory may add its own sharpening parameter"** (the audit row above) | adding `beta` to `rsa_l1_salience` reproduces the free-temperature ridge on pragmods (alpha 0.21, beta 2.86): what the audit called a sharpening alpha cannot mimic was, inside one theory, the same ridge again. A theory can still claim sharper listeners through its own mechanism |
| Refits and fold bounds | A fit that fails the convergence gate is refitted only as a near miss (main's rule: R-hat ≤ 1.2, bulk ESS ≥ 20, ≤ 2% divergent). Each cross-validation fold of a theory in play is limited to 3× that theory's full fit (at least 10 min; 90 min for a fit recorded without its time). A failed fold leaves the theory unscored and untrusted, on record (`cv_problem` in `history.json`, and in the theorists' brief) | run B's fifth fold of `feature_parsimony_listener` ran 5 h 50 min to R-hat 1.55, then began an unlimited refit while the round waited. A warm-started fold runs 800 iterations to a full fit's 2,000; past 3× it is stuck, not slow |
| ~~Run B′ instead of the rest of run B~~ superseded 2026-10-24 (below) | ~~Run B stops once experiment 1's rounds are brought back; run B′ runs 2 experiments on the fixed code~~ | the stop never reached Sherlock; run B ran on |

## Decisions (PI, 2026-10-24)

| Decision | Choice | Why |
|---|---|---|
| A corpus-relative admission limit | A candidate's admission fit is limited at 2× the median full-fit time of the loop's starting theories (seeds or carried, fitted at setup), never below 30 min (`orchestrator.ADMISSION_FACTOR`). Fixed for the loop, so it does not ratchet up as slow candidates are admitted. Recorded in `history.json`, stated in the theorists' brief, and passed to their self-check, which fails a projection over it. Fits cached without their time cannot set it: the loop refuses and asks for a fresh root | run B, left running, finished experiments 1 and 2 with 43 of 61 candidates too slow. The temperature audit shows why: on all of E, 12 of the 15 bar theories take longer than 30 min themselves (`rsa_l1` 26 min, `rsa_l2` 46, the starting incumbent 83), so a refinement of the incumbent could never be admitted. The 30 min was the closed campaign's, set for smaller corpora; ~90 min on all of E keeps its intent (a candidate may cost a few times a typical theory) |
| The designer's clock and the harness's submission (from run B's report) | Each designer session's deadline is in its brief (a clock time) and at the end of every tool's output, with an urging note in the last 30 min. A session that ends without a submission is submitted for by the harness: its spec with the best full `eig` meeting the floor (`tools.best_scored_spec`), checked like any submission and recorded `submitted_by: harness` in `record.json` | experiment 3's first designer spent 92 of its 120 min on its own scripts, had a full eig of a spec at 0.0284 bits/$ against a 0.0122 floor at minute 119, and submitted nothing; one more such cycle would have stopped the chain. The harness's checks (re-score, link check, pilot) stay the gate either way |
| Catch trials built in open code (bug) | `screens.catch_scene` replaces the import of the closed `src.rsa.experiment.design._catch_spec`, drawing identically (tested) | the agents' tree leaves `src/rsa/experiment/` out, so the designer's `validate` crashed in all four of run B's designer sessions; each worked around it by writing its own stub into a `sitecustomize.py` in its private home. Its reports then rested partly on agent-written code. The frozen designs did not: the harness checks with the real code |
| Let run B finish | Run B runs to its end (experiment 3) as the old code's baseline; experiments 1 and 2 are brought back now | it was past experiment 2 when the stop would have reached it |
| Run B′ is a replay of run B's experiment 1 (proposed by the driver, awaiting the PI) | Same frozen design, simulated data and starting theories as run B's experiment 1; only its loop reruns, 3 rounds of 6, on the new code (`REPLAY_FROM` in `open_rehearsal.sbatch`) | a before/after on identical data isolates what the code changed (too-slow rate, admissions, incumbent changes) for a third of a two-experiment B′'s cost; run B already exercised the design, collection and carry-forward machinery three times |

## Decisions (PI, 2026-10-25)

| Decision | Choice | Why |
|---|---|---|
| An upgraded closed arm, supplemental | The closed campaign's nine inner loops rerun on its own frozen designs and live data with the open campaign's theorist tooling. Their finals join the bakeoff as a supplemental comparison (`PREREG_DRAFT.md` §5). Run only if the open campaign completes and proceeds to the bakeoff, decided before any bakeoff data | the sampler report and the identification prompt help the open theorists, and the prompt bears on claim 2; open against upgraded closed isolates the designer and the changing paradigms. Making the arm contingent on completion, not on any result, keeps the decision free of the outcomes |

## Decisions (PI, 2026-10-26)

| Decision | Choice | Why |
|---|---|---|
| One harness, two arms | The closed campaign's approach becomes the **fixed arm** of this harness: the closed forced-choice paradigm as one frozen spec, its displays chosen by the same EIG tool, no designer. Everything else is shared by construction (`PREREG_DRAFT.md` §2). This supersedes the upgraded closed arm of 2026-10-25 | the open harness had drifted from the closed code (theorist tooling, reference keeping, the admission limit, budget, agents); one harness makes the designer the only difference |
| 4 chains per arm, behind a feasibility gate | Open chain 1 runs first, live, with pre-registered pass criteria (`PREREG_DRAFT.md` §4). On a pass, open chains 2–4 and fixed chains 1–4 run, concurrently and interleaved. Registration comes before chain 1 | the method claim's unit is the chain; 3 against 3 was thin. The gate spends on 7 more chains only once the open arm is shown to run with real people |
| A larger per-experiment budget, for every experiment of both arms | Amount to be set before registration | run B's experiments 2 and 3 barely separated the leading theories at N = 200 |
| Gemini 4 for every agent | Designer, theorists and critic, in both arms | both arms share the harness, so the upgrade is not an asymmetry |
| Keep each earlier paradigm's best theory | It is never pruned or retired by the cap while it is best there, so the 4-SE guard keeps its reference | run B′ pruned E's best theory (about 110 nats ahead on E) on the newest block alone, after which the guard compared only survivors |
| 4 chains × 4 experiments per arm; 2× budget; chain 4 repeats chain 0's seeds; the gate's thresholds as drafted (PI, later the same day) | $266.67 per experiment. The gate judges open chain 1's first 3 experiments; its 4th runs on a pass | depth tests generality across more operationalizations and gives the designer room; 4 chains keep the chain-level consistency argument (no chain-level test is strong at this size, so the primary inference stays in the bakeoff). About $13k before the bakeoff (a projection) |
| Two-measure paradigms, if not too complex | A paradigm may ask a second question about the same screen, each measure with its own link and its own block. The joint outcome space is capped for the EIG | it tests whether one theory explains two responses from the same people |
| PI review: the first paradigm of each open chain only | As before | |
| Infrastructure, spread across nodes; failure rules pre-registered; a frozen code tag; the Gemini 4 model ID pinned and recorded; claim 3's metric in the pre-registration; live monitoring of every response type | Gemini spend is not capped (credits) | |
| Subjective randomness: not now | No open branch for subjective randomness until after the RSA feasibility chain | scope |

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

**No hand ports (2026-10-22).** From 10-21 to 10-22 the one model with its own decision
temperature (`softmax_surprisal_isolated_costly_l4`'s `beta`) was ported by hand with `beta`
cut, left to the link. Since E's link temperature is fixed at 1 (Decisions, 2026-10-22), such a
parameter is the theory's own sharpness, shared by every paradigm, and the model ports by rule
(a) with `beta` kept.

**Composing** a theory with a link gives an ordinary model file (`compose`): the theory, then
the link's parameters and its `choice_probs`. The existing fitting, CV and loop code runs it
unchanged.

**The 20 models** (5 starting models, 12 promoted seeds, 3 closed finals; `tests/test_rsa_open_port.py`):
- 19 port by rule (a), `softmax_surprisal_isolated_costly_l4` with its own `beta` kept;
- chain 2's final (`competition_endpoint_binding_lapse_l2`) ports by rule (b). Its lapse
  mixes in its own salience prior, on word trials only.

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
     inside, as now), and reports EIG per participant-minute. The chain driver
     (`src/rsa_open/chain.py`, `ChainConfig.bar_dir`) fits the bar to the data so far even after
     the loop has pruned it, and the brief labels a pruned bar theory as such. Without that, a
     loop that pruned down to one theory left the next design zero bits and no paradigm above the
     floor; the first two-experiment dry run hit exactly this;
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
  selection, then the joint EIG and power on fresh scenarios. The open campaign runs its greedy
  search lazily, as main's design does (`design.LAZY_FULL_EVERY`: a full pass every 16 picks, in
  between only the candidates whose last gain could still win; the closed campaign's designs stay
  exact). On the smoke run's context (4,940 units, 40 picks): 142 s instead of 820–860 s, joint
  EIG 1.912 and 1.914 bits (two seeds) against exact greedy's 1.913.
- **The floor** is half the bits per dollar of the closed paradigm (1,597 scenes, 200 × 13 of
  52, the choice link) under the same theories.
- **Sliders** (the ordered-beta link) are scored on their responses binned into 12 outcomes
  (0, ten interior bins, max). Coarsening can only lose information, so the score is a lower
  bound. **Bets** (allocation, built 2026-10-21) are scored as a sub-bet of at most 10 points
  with at most 300 splits (`links.design_bet_total`: 10 points for up to 4 objects, 6 for 5, 5
  for 6). A random part of a Dirichlet-multinomial bet is a smaller one (a Pólya urn; tested by
  simulation), so the sub-bet is a garbling of the bet and its score again a lower bound.

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
| legibility (SVG), by 3 fresh chats: each base named freely (a name other than the domain's goes to a judge chat); the part each single feature adds, among the domain's words; which objects of 6 real test trials a word is true of (exact) | ≥ 90% right, no feature or trial misread by half (an unrecognised base: a warning) |

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

**The critic** (`src/rsa_open/loop/critique.py`, built 2026-10-21): the closed critic's rules
(8 statistics, 1,000 replicates, p ≤ 0.05 with BH q, one retry, "no critique" recorded) on the
newest paradigm's trials, the rows selection ranks on. Each replicate redraws only the response,
through that paradigm's frozen link, from one posterior draw of the incumbent's theory and link
parameters; the trial's design (a probed object, a target, a judged message) stays as presented.
1,000 replicates of 2,400 trials take 20–30 s.

## 6. Parity with the closed campaign

The same seeds and chain split, the same literature data, the same theorist model and sandbox,
the same rounds and slots, 3 chains × 3 experiments, the same per-experiment budget, and no
closed-campaign data. What differs: the designer, the page built from its spec, and that the
paradigm, and with it the link, changes every experiment.

**Asymmetries added since (2026-10-23/24):**
- **The admission limit is corpus-relative**: 2× the median starting fit, never below 30 min.
  Argued as parity of constraint, since a fixed 30 min on all of E rejected 77% of run B's
  candidates where the closed campaign's rejected about a third. The two campaigns' too-slow
  rates are reported side by side.
- **Theorist tooling the closed campaign never had:** the sampler report, the identification
  prompt, near-miss refits and bounded folds. **The upgraded closed arm** (`PREREG_DRAFT.md` §5)
  removes this confound: the closed campaign's nine loops rerun on its own data with this tooling.
  It is a supplemental comparison, run only if the open campaign completes and goes to the
  bakeoff (PI 2026-10-25).

## 7. The final bakeoff (pre-registered, as `rsa-prereg-final` was)

**Competitors:** the three open finals and the three closed finals (ported, §3), plus the seeds;
supplementally, the three upgraded closed finals (`PREREG_DRAFT.md` §5).

**Draft:** `PREREG_DRAFT.md`.

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
| 1 | Paradigm-spec schema and validator; link library with recovery tests; theory + link refactor; port the seeds and closed finals (exact-equivalence tests) | cloud | **done** (`src/rsa_open/`): all 20 models port (19 by rule a, chain 2's final by rule b; one by hand until 2026-10-22) and equal their originals; five links (choice, rating, allocation, production, binary judgment), each with a temperature and a lapse, each passing the recovery test; every theory has a speaker; the paradigm spec and its validator. Not yet: freezing the ported closed finals (with their derived speakers) before the campaign |
| 2 | Designer tools as an agent CLI; designer brief; the page; simulated pilot | cloud | **built** (§4.1–4.3): the EIG tool; the tools CLI (`validate`, `eig`, `link-check`, `submit`, 40 calls); spec-driven trial lists and one page for all six response types (tested in Chromium); the pilot with Gemini 3.8 Flash participants and its gate (a live 3-participant run passed); the designer cycle (brief, the harness's own re-checks, up to 3 pilots, one retry, freeze); a real Gemini 4 designer cycle froze a paradigm (§4.3); EIG for bets (2026-10-21). Not yet: designer-proposed links |
| 2b | Fitting across paradigms (shared theory parameters, each paradigm its own link parameters, E included); the open inner loop (theorists; grouped CV within paradigm; the k = 4 SE accountability rule; carry-forward); the chain driver (context → designer → frozen paradigm → simulated collection → inner loop → timing record); rehearsal ground truths; Sherlock jobs | cloud | **done (2026-10-22)**: everything above, and run A, the trimmed dry run with real agents (pragmods only, 1 experiment, 1 round of 3), passed end to end in 4 h 20 min for $9.28 of agent spend (`DRY_RUN_A_REPORT.md`, `data/rsa_open/dry/`). The designer froze a bet paradigm on its first submission (3.63 bits, 0.032 bits/$ against a 0.012 floor; pilot and legibility 100%). The critic's 8 statistics all flagged the incumbent on duplicate and crowded displays, the hidden truth's mechanisms, and the round's new incumbent, `rsa_l2_duplicate_suppression_l0`, was discovered there. No design theory was screened out; no agent file names the truth |
| 3 | Rehearsal on simulated participants with a ground-truth theory across paradigms | Sherlock | **next**: run B (`HANDOFF_rehearsal.md` §3), chain 0, 3 experiments on all of E |
| 3b | The integrated harness (checklist below); a simulated rehearsal of both arms; registration (`PREREG_DRAFT.md`) | cloud, Sherlock | **next** (2026-10-26) |
| 4 | The feasibility chain: open chain 1, live, 3 experiments, PI review of its first paradigm; the gate (`PREREG_DRAFT.md` §4) | live | |
| 5 | On a pass: open chains 2–4 and fixed chains 1–4, concurrently | live | |
| 6 | Bakeoff: arms F and T (second registration) | live | |

**Phase 3b checklist (2026-10-26).**
1. Reference keeping: each earlier paradigm's best theory is never pruned or retired.
2. Gemini 4 for theorists and critic; the model ID pinned and recorded on every call.
3. The 2× budget; the fixed paradigm's reference at 400 people; 4 experiments; chain 4's seeds.
4. **The fixed arm** (built 2026-10-26):
   - `src/rsa_open/fixed.py` writes `fixed_paradigm.json`: the closed campaign's six pragmods items
     (pictures and words), each object drawn on one of an item's three base pictures (the spec's
     new base `variants`); Bob's wording; the closed scene space; 13 + 2 trials, $0.50 for 3 minutes,
     for 400 people.
   - The pictures are embedded PNGs, which the validator accepts only when their bytes are one of the
     curated pragmods files. Those files are not in the agents' tree, so a designer cannot use them.
   - It runs on the open page, so it gets the same consent, comprehension checks, practice and
     catch trials as any open paradigm. Two changes from the closed page: two comprehension
     checks (the closed page had none), and "Click on the one you think Bob is talking about" in
     place of naming the item.
   - The harness's pictures skip the legibility check, which exists for a designer's drawings.
   - A domain now needs only the features of the scenes it can draw (the page already rotates
     among eligible domains): pizza, boat and Christmas tree have 3 features, as in the closed
     campaign.
   - The chain mode (`arm="fixed"`) sends the spec through the harness's own checks (EIG re-score
     under the design theories, link check, page, pilot), with seeds salted by chain and experiment.
     A failure raises `FixedDesignFailed`. `ARM=fixed` in `open_rehearsal.sbatch` runs cell
     `fixed_chain_<k>` beside the open arm's, sharing the work root and its ground truth.
5. **Two-measure paradigms** (built 2026-10-26):
   - A spec may add `second_response`, a second question about the same screen, asked once the
     first is answered. After a listener response it is a choice or a rating of at most 7 points;
     after a speaker response, a yes/no judgment.
   - The **pair link** (`links.PairLikelihood`) reads one side of the theory. The two responses
     are independent given it, so its log-likelihood is the sum of the parts'. The second part's
     parameters are prefixed `q2_` (its own temperature and lapse), its design fields `second_*`,
     and its response sits in the first's as `"second"`. One block per experiment, as before, so
     standing, the guard and pruning are unchanged.
   - **The design tool** scores every pairing of the two measures' units per scene, with the
     product of their outcomes, capped at 300 joint outcomes.
   - **The page** draws the second question as a second step of the same trial (tested in
     Chromium). The pilot asks it as a second question about the same picture; catch trials judge
     the first measure; timing adds both answers.
   - **The critic** sees the second response as `second_*` columns and simulates both.
   - The designer's brief describes the option; a theory may not use a `q2_` parameter name.
6. **Live collection for open chains** (built 2026-10-26, `src/rsa_open/live.py`; tests in
   `tests/test_rsa_open_live.py`, one of them in Chromium against stand-in `/assign` and `/submit`):
   - The page is the open player with the closed campaign's live hooks (the list from `/assign`, the
     data to `/submit`, Prolific's completion URL). The deployment injects the IRB consent gate.
   - The deployment is main's, under the closed campaign's Prolific project (`rsa_reference`), so the
     project-scoped blocklist excludes everyone from any earlier RSA study, closed or open, and
     admits the subjective-randomness studies' people. One study per experiment (main's guard).
   - Each study's N, pay and minutes are its paradigm's (`recruitment_overrides`, a new optional
     argument of main's deployment), under one neutral name and description for both arms
     (`scripts/rsa/live/open_live.yaml`), so a listing does not say which arm it is.
   - **One study recruits at a time across the campaign** (a lock every chain's job shares):
     Prolific applies the blocklist at publish, so two open studies could share a person. Chains
     publish in the order they are ready, which interleaves the arms.
   - Participant ids are the campaign's (`p<k>`, one private map). Someone seen in an earlier study
     is excluded, on record; raw responses stay private on Sherlock.
   - **Exclusions:** a missed catch trial (the closed campaign's rule), failed comprehension checks
     on the second reading, a repeat person, an incomplete record (`PREREG_DRAFT.md` §4b).
   - **Monitoring:** every 10 minutes while a study fills, and at its end: exclusions by reason,
     comprehension, catch trials, people giving the same answer on nearly every test trial (a
     screen position for a choice, the same rating, slider value, allocation, word or yes/no, both
     parts of a two-measure trial), fast responders, minutes. `python -m src.rsa_open.monitor`
     prints it for a running study.
   - **The PI's review** (`python -m src.rsa_open.review`): an open chain's experiment 1 is published
     only with an approval naming its paradigm's sha256. A test launch (deploy + Prolific draft)
     needs none: it is how the PI sees the page.
   - **A smoke test** fields another run's frozen design at a few people and stops; it is never
     modelled.
7. **Infrastructure** (built 2026-10-26): `scripts/rsa/live/open_launch.sh` submits a ladder of
   jobs per chain (`open_live.sbatch`; default 4 × 48 h on `normal`, each after the last). A job
   never starts a design or loop it cannot finish (8 h and 24 h left, `STAGE_MIN_HOURS`); it hands
   the chain to the next job, which resumes it. A job that finishes, stops for review, made a test
   deploy, or failed writes the cell's STOP; the ladder's later jobs then exit, and a person
   relaunches with `CLEAR_STOP=1` (`PREREG_DRAFT.md` §4a: infrastructure resumes, a failure is
   looked at first). A live chain runs only on the config's frozen `code_tag`, clean, staged once
   (`stage_code.sh`; later fixes only as logged re-stages). The model ID is on every token-usage
   record. **Still open:** the schedule (below, after run C) and the tag itself (at registration).
8. The pre-registration's remaining pieces: claim 3's metric, chain-level reporting and failure
   rules.
9. The rehearsal of both arms, simulated, with Gemini 4 and the ground-truth check. **Proposed
   (2026-10-26):** not a third full-size rehearsal, since runs B and B′ tested the loop at size. Run C
   (plumbing) runs three cells at once on run A's trimmed loop, each on its own node: an open chain
   (2 experiments), a fixed chain (2), and the fixed paradigm with a second measure
   (`fixed_pair_paradigm.json`, 1; a designer picks its own format, so the two-measure path is
   rehearsed on a harness spec). It includes one kill-and-resume and the ground-truth check.
   Expect half a day to a day and about $30. Run D tests the live path: a test deploy of each arm
   (no money), then, with the PI's go-ahead, about 15 real people each (about $25, never
   modelled). `HANDOFF_rehearsal.md` has the commands.

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
