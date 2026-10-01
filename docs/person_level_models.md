# Person-level models in the pipeline

On human data the leading models have **person-specific parameters**: in the
October 2026 live run every one of the models within 4·dse of the best after
run 1's experiment 1 had them (an ideal switching rate, a balance weight, a
streak aversion and a lapse rate per person). This page says how each stage
treats a model that binds `participant_id`, what such a model must do, and
what is planned so that the treatment carries over to future experiments and
projects.

## The model's claim about new people

A person-level model is hierarchical: person *i*'s parameters are
θᵢ = f(μ + σ·zᵢ) with zᵢ ~ Normal(0, 1), beside shared parameters φ (β, say).
MCMC fits μ, σ, φ and every zᵢ jointly: each zᵢ from that person's trials,
shrunk towards the population; μ and σ from everyone. The model's claim about
people in general is the mechanism plus the population distribution
G(θ | μ, σ). Its prediction for a **new** person marginalizes θ over G, with
the hyperparameters from the posterior:

    p(y | pair, data) = ∫∫ p(y | pair, θ, φ) · G(θ | ψ) · p(ψ, φ | data) dθ d(ψ, φ),   ψ = (μ, σ)

A slot of a per-person vector that no participant's data reached is, in every
posterior draw, a draw from G(ψ) (its z keeps its Normal(0, 1) prior), so
predicting a pair "as" an unobserved participant id samples this integral by
Monte Carlo without knowing which variables of an agent-written model are the
hyperparameters.

## Stage by stage

| stage | a model that binds `participant_id` | where |
|---|---|---|
| fitting | person and population parameters fitted jointly | `pymc_inference.fit_model` |
| scoring, best model, pruning | ELPD-LOO leaves out one **trial**: the held-out trial's person still contributes their other trials, so it measures predicting a known person's next choice. This rewards person-level flexibility and does not measure generalization to new people (plan 3) | `scoring.py`, `model_zoo._prune_losers` |
| novelty gate | posterior-mean `p_left` on the novelty pool, averaged over the **training** participants: an empirical stand-in for G (plan 2) | `model_zoo._pool_prediction` |
| design after data (experiments ≥ 2) | predicted as `DESIGN_NEW_PARTICIPANTS` (40) **new** participants, ids past every training id, and averaged draw by draw: the integral above (since 2026-09-30) | `eig._new_participant_draws` |
| prior design (experiment 1) | screened out, on record (plan 1) | `eig._screen_usable_models` |
| recovery evaluation (holdout harness) | averaged over the training participants (plan 2) | `holdout_eval._participant_ids_in` |

In the design, new participants answer one pair independently, so each
pair's Binomial(n, p̄) in the EIG is exact under the model; one person's
answers to different pairs are correlated, which the joint EIG ignores (plan
5). With 40 new ids the Monte Carlo error of p̄ matches that of averaging over
a 40-person sample, at the same cost (40 prediction passes per model; about
1-2 minutes over the 43,434-pair pool).

## What a person-level model must do

The candidate prompt (`src/pipelines/inner_loop/prompts/pymc_theory.md`,
"Participant effects") tells agents this; the design enforces it:

- bind `participant_id` directly and index the per-person parameters by it;
  never renumber participants inside a hook (`_require_own_slot`: a hook that
  does not pass the id through unchanged is screened out of the design);
- give the per-person parameters a population distribution, and size them
  well beyond the participants so far (`shape=400`): ids keep counting across
  a run's experiments (run-unique, `run_unique_participant_ids`; 40 per
  experiment reach 119 by experiment 3), and the design needs slots past them
  (`NoNewParticipant`: an index error for a new id while a training id
  predicts screens the model out, with the reason, in `screened_out.json`).

A model sized to the participants so far also cannot be fitted on the next
experiment's data (its new ids fall outside the vector), so spare slots are
needed to carry a person-level model at all.

## Plans

Not implemented yet; in rough order of need.

1. **Prior design over person-level models.** Before any data every slot is a
   prior draw, so experiment 1 could predict such a model as new ids 0…K−1
   from the prior predictive, letting a project start from person-level
   models. Change: the prior path of `design_exhaustive` and the two
   `tests/test_eig_pymc.py` tests that pin the drop.
2. **One definition of a new participant.** Move the novelty gate and the
   recovery evaluation from the training-participant average to the same
   unobserved-slot marginalization as the design. Deferred until the live
   series ends: it would change novelty verdicts mid-series.
3. **Model comparison on new people.** Report, per experiment, a new-subject
   ELPD: each participant's trials scored under a fresh draw from the
   population (an unobserved slot), close to leave-one-participant-out (each
   participant also informed μ and σ). Then decide whether pruning should use
   it: the October 2026 run pruned at 4·dse because trial-level differences
   between person-level models are measured very precisely.
4. **A structural slot check at admission.** Read each per-person vector's
   length and reject a candidate that leaves no room for the next
   experiment's ids, with the reason, instead of discovering it at the next
   design or fit.
5. **Within-person correlation in the joint EIG.** Simulate scenarios per
   participant (one θ answering every picked pair) rather than per pair.
   Heavier; only if designs show that it matters.
