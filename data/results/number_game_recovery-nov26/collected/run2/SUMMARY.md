# Live (human) outer-loop — results summary

- Source run root: `/sailhome/wurgaft/auto-psych-runs/nov26b-collect/assembled/collect_src/run2`
- Runs: 3 (bigelow_piantadosi_independent, bigelow_piantadosi_mixture, tenenbaum_rules_similarity)
- Experiments per run: experiment1, experiment2, experiment3
- Prolific worker/study ids scrubbed (no PII committed).

## Winning model per (run, experiment)

| run | experiment | best model | P(best) | participants | trials | runner-up | Δelpd | dse |
|---|---|---|---|---|---|---|---|---|
| bigelow_piantadosi_independent | experiment1 | familiarity_relaxed_size_principle | 0.570 | None | 2250 | capacity_limited_evidence_integration | 0.61 | 3.39 |
| bigelow_piantadosi_independent | experiment2 | parity_veto_sparse_familiar_rules | 0.826 | None | 4500 | rule_or_range_kind_commitment | 1.67 | 3.34 |
| bigelow_piantadosi_independent | experiment3 | final_digit_glance_effortful_search | 0.880 | None | 6750 | retrieval_vs_checking_rule_search | 1.99 | 9.58 |
| bigelow_piantadosi_mixture | experiment1 | without_replacement_free_outlier_rate | 0.208 | None | 2250 | shown_examples_not_repeated | 0.26 | 1.67 |
| bigelow_piantadosi_mixture | experiment2 | margin_and_heading_range_prior | 0.238 | None | 4500 | first_example_marks_range_edge | 0.16 | 1.03 |
| bigelow_piantadosi_mixture | experiment3 | position_decaying_example_evidence | 0.256 | None | 6750 | position_decay_upward_room | 0.10 | 0.77 |
| tenenbaum_rules_similarity | experiment1 | without_replacement_example_sampling | 1.000 | None | 2250 | None | n/a | n/a |
| tenenbaum_rules_similarity | experiment2 | no_digit_containment_span_concepts | 1.000 | None | 4500 | None | n/a | n/a |
| tenenbaum_rules_similarity | experiment3 | new_set_criterion_ratio_ceiling | 0.261 | None | 6750 | graded_ratio_ceiling_criterion | 0.39 | 0.59 |

## Winning-model agreement across runs

- experiment1: agreement=no → familiarity_relaxed_size_principle, without_replacement_example_sampling, without_replacement_free_outlier_rate
- experiment2: agreement=no → margin_and_heading_range_prior, no_digit_containment_span_concepts, parity_veto_sparse_familiar_rules
- experiment3: agreement=no → final_digit_glance_effortful_search, new_set_criterion_ratio_ceiling, position_decaying_example_evidence
