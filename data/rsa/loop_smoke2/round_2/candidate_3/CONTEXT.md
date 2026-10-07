# Context

- Candidate directory (write your three files here): `data/rsa/loop_smoke2/round_2/candidate_3`
- Responses (read-only): `data/rsa/loop_smoke2/responses.csv`: 50587 forced-choice trials,
  one row per click, from experiments E10_oddman, E1_dv, E2_manip_check, E3_ling_frame, E4_prior_frame, E5_baserate, E6_valence, E7_color, E8_levels, E9_twins, color_prior_rerun, levels_prior_action, md2022_main, md2022_pilot, md2023_e1_replication, md2023_e2_remapped, md2023_e3_all_messages, md2023_e4_shapes, md2026_shapes, sequences, sikos2021_e1, sikos2021_e2, sikos2021_e3, size, speakers. Columns
  include `experiment`, `condition`, `objects` (JSON list of 0/1 feature
  lists, one per object), `feature_names`, `query` (`utterance` or
  `prior`), `utterance` (feature index heard), `familiarization`,
  `grayscale`, `framing`, `choice` (object index clicked), `participant_id`
  and `trial_index`. You may analyse it (e.g. with pandas, from the shell) to
  find what the current models miss; your model file may not read it.
- Inner-loop round 2 of 2.
- Self-check: `/home/user/auto-psych/.venv/bin/python3 -m src.rsa.loop.check_candidate data/rsa/loop_smoke2/round_2/candidate_3 --responses data/rsa/loop_smoke2/responses.csv`
  It takes 2-10 minutes (it compiles and fits your model). Give the shell
  tool a timeout of 900000 milliseconds for it (the tool's
  `timeout` parameter); with the default of 120 s the check is killed before it
  prints anything. Run it once and read its result; do not re-run it unchanged.
- Write and test your files **in your candidate directory only**: no drafts in
  `/tmp`, the repository root or anywhere else. The loop reads only
  `data/rsa/loop_smoke2/round_2/candidate_3`; a candidate.py anywhere else counts as no candidate.
