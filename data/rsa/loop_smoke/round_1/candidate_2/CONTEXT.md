# Context

- Candidate directory (write your three files here): `data/rsa/loop_smoke/round_1/candidate_2`
- Responses (read-only): `data/rsa/loop_smoke/responses.csv`: 6703 forced-choice trials,
  one row per click, from experiments E10_oddman, E1_dv, E2_manip_check, E3_ling_frame, E4_prior_frame, E5_baserate, E6_valence, E7_color, E8_levels, E9_twins, color_prior_rerun, levels_prior_action, sequences, size, speakers. Columns
  include `experiment`, `condition`, `objects` (JSON list of 0/1 feature
  lists, one per object), `feature_names`, `query` (`utterance` or
  `prior`), `utterance` (feature index heard), `familiarization`,
  `grayscale`, `framing`, `choice` (object index clicked), `participant_id`
  and `trial_index`. You may analyse it (e.g. with pandas, from the shell) to
  find what the current models miss; your model file may not read it.
- Inner-loop round 1 of 1.
- Self-check: `uv run python -m src.rsa.loop.check_candidate data/rsa/loop_smoke/round_1/candidate_2 --responses data/rsa/loop_smoke/responses.csv`
