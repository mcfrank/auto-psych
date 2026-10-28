---
name: volatile-model-speed
description: How to keep volatile-changepoint concept models fast enough for the 30-min admission fit
metadata:
  type: project
---
Volatile changepoint models (sequential update over 42 rules + interval groups) are near the 30-min admission limit; the machine is heavily loaded (1 CPU visible, load ~25).
Tricks that helped (2026-10-21, experiment3 candidate_5_repair_1): bucket sets by example count so each bucket runs only its own updates; drop the noise log-lik constant (it cancels on normalisation); avoid adding parameters that trade off (e.g. range inflation + range tempering).
**Why:** volatile_split_rule_range_inflation was rejected as too slow. Its repair took depth ~5 (31 steps), about 6 min per chain at 200 tune / 250 draws.
**How to apply:** benchmark with src.models.pymc_inference._warm_start_sample at target_accept 0.99 before finishing.

Timing reference (2026-10-21, experiment3 iter_1 candidate_1): the non-volatile incumbent inflated_size_familiar_rule_prior plus one parameter took ~14 min wall time for check_candidate (100 tune / 100 draws, 1 chain), compile included, on the loaded machine. Use background runs.

Non-volatile models built on inflated_size_familiar_rule_prior are slow too: the check_candidate smoke fit (100 tune + 100 draws, 1 chain) took ~340 s per run on 2026-10-21 (load ~9), and the full check takes ~12 min. So run the check in the background with a long timeout and redirect its output to a file, not `| tail`. inferred_stray_example_rate on that base was dropped as too slow in a fold fit.

Repair strategy (2026-10-21, experiment3 iter_1 candidate_5_repair_1, final attempt): volatile_final_digit_union_concepts (volatile base + 375 digit-union concepts) timed out at 30 min. The repair moved the same union mechanism onto the non-volatile inflated_concept_size_principle base, as final_digit_union_inflated_size, with unions collapsed into at most 30 groups per set by (arity, examples inside, anchor inside). On a final repair, prefer a non-volatile base.

Note on novelty (2026-10-21, experiment3 iter_1 candidate_3): splitting the incumbent's size inflation into separate rule and range inflations was rejected as a near-duplicate (p_yes RMSE 0.0013 < 0.002). The data do not separate those two inflations, so tweaks of that kind do not count as new. Final-digit unions, rule-check failures and surface-numeral fallback were all taken by other slots the same round. trial_index cannot be separated from stimulus, because each (set, target) cell sits at a single trial index.

Exploratory idea used (2026-10-21, experiment3 iter_2 candidate_0): set_size_triggered_rule_search. It is the incumbent with the prior share of rules (p_mix) given a logit slope over set size, centred at 3 examples. It adds one parameter, so the cost stays close to the incumbent's.

Idea log (2026-10-21, experiment3 iter_2 candidate_2): schema_grammar_rule_prior. It replaces the incumbent's 42 Dirichlet rule weights with 8 schema weights split equally within each schema, times size**exponent. A simpler, faster, high-variance test of whether rule-specific familiarity matters.

Refinement used (2026-10-21, experiment3 iter_2 candidate_3): serial_position_ends_inflated_size. It is the incumbent plus a last-example evidence multiplier (recency_weight), applied only to sets of 2+, with interval groups also split by whether the last example is inside. Data note: only one set ("8,16" vs "16,8") appears in two orders, so order effects show up mainly as position-within-set weighting. Pitfall: when splicing the incumbent source, split on the module docstring only (src.split(DOCQUOTE+"\n", 1)[1]). concept_sets() has its own docstring; maxsplit 2 silently drops code and gives "forbidden import: <unparseable source>" (hit again iter_3 candidate_5).

Timing (2026-10-21, experiment3 iter_2 candidate_5): conservative_updating_inflated_size (the conservative linear-pool update on the volatile_changepoint_inflated_size code base) passed check_candidate. Sampling took 296 s and the whole check about 14 min wall time.

Cell-level trick (2026-10-21, experiment3 iter_2 candidate_1_repair_1): the model_loop data have 6750 trials but only 1312 distinct (set, target) cells and 180 sets. Compute interval membership sums and p over cells, then set p_yes = p_cell[row_cell]. This cuts the dominant (trials x ~446 groups) operations about 5x. queried_target_self_evidence took 697 s per smoke fit before this change and was rejected as too slow.

Per-cell trick (2026-10-21, experiment3 iter_2 candidate_0_repair_1): the incumbent-family code builds `group_in_concept` as (trials x groups), 6750 rows. Computing predictions per distinct (set, target) cell instead (1312 cells: `cell_set`, `cell_group_share`, `cell_rule_in`, then `[row_cell]` to trials) cut the logp+grad time from 0.054 s to 0.018 s with identical logp. Apply it to any model copied from inflated_size_familiar_rule_prior. Note that 180 distinct sets exist; a naive `cut -d,` count is wrong because of the quoted commas.

Speed fix (2026-10-21, experiment3 iter_2): all six iter_2 candidates timed out at 30 min (load was ~38). Most were incumbent + 1 parameter. In the candidate_4_repair_1 rewrite, p_yes was computed per distinct (set, target) cell, 1312 cells versus 6750 trials, then indexed by row_cell. Also, mask/remaining-size arrays were precomputed instead of using pt.switch loops. Together these cut the gradient eval to ~1/3 of the incumbent's. Apply this cell trick to any model on this base.

Speed fix for the non-volatile base (2026-10-21, experiment3 iter_2 candidate_3_repair_1): the data have only 1312 distinct (set, target) cells across 6750 trials, and at most 39 targets are asked per set. Replace the per-trial `group_in_concept` (trials x groups) with share tensors padded to the asked targets, (sets, groups, 39) and (sets, rules, 39). Compute `pt.batched_dot(posterior, share)` once per set, then index trials by row_cell. This brought logp+grad from 53 ms to 12 ms with identical MAP. Elementwise multiply-and-sum was slower than batched_dot, and per-cell gathers were also slower. no_repeat_example_sampling and serial_position_ends_inflated_size both timed out at target_accept 0.99 on the old layout. Profile pattern: /tmp/c3r/prof.py (may be gone).
Result: check_candidate passed with OK. The smoke fit sampled in ~125 s per run, versus ~340 s before; whole check ~10 min.

Timeout repair (2026-10-21, experiment3 iter_2 candidate_5_repair_1): conservative_updating_inflated_size timed out at admission even though its smoke check had passed. A warm-start benchmark at target_accept 0.99 (150/150 draws) showed the same depth as the incumbent (~31 steps). The fix had two parts:
- Better-identified parameters: one shared evidence weight, since the rule and range weights were nearly equal, and a LogNormal(log 3, 0.75) prior on size_inflation instead of HalfNormal.
- Cheaper gradients: sort sets by size, largest first, and run each update step k only on log_post[:n_active[k]] with pt.set_subtensor. This cut one gradient from 0.067 s to 0.047 s; the incumbent's is ~0.053 s.

The benchmark script is at /tmp/c5r/bench.py (it may be gone). It must run from the repo cwd and not with -I, or pyprojroot fails.
Result: the per-cell set_size_triggered_rule_search ran warm-start NUTS at target_accept 0.99 (200 tune / 250 draws, 1 chain) in 278 s, tree depth 5, 0 divergences. Its check_candidate smoke fit took 121 s, against about 340 s before. The posterior slope was 0.07 ± 0.19, near zero, so it is a novelty risk.
