---
name: times-table-rule-sampling
description: exp2 iter2 c1 exploratory: one-sampled-rule model (parity, x3-x12 tables, digit, specials vs soft region), smoke -2868
metadata:
  type: project
---
Exp2 iter_2 candidate_1 (exploratory slot): `times_table_rule_sampling`. Each trial samples one property shared by all examples (type salience + table-size slope + accumulation*(n-1), vs soft hull "region" rule), lapse to guess base. 9 params, prepare_observed with (N,25) consistency/membership matrices; fit ~15 s/100 draws.
Smoke log-lik -2868 vs incumbent multiple_route_exemplar ELPD -2721 — ~150 nats behind; rule-sampling family still well behind exemplar similarity (cf. [[log-odds-distorted-concept-learning]]).
Targets critique stats: single-example factor-kin, small-vs-large example multiples, interior fill ([[exp2-iter2-critique-stats]]).
