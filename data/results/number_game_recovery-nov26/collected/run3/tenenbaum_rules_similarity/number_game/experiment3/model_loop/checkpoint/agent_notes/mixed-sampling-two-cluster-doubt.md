---
name: mixed-sampling-two-cluster-doubt
description: exp3 iter_2 c5 repair: theta mixed sampling grafted onto two_cluster_kind_doubt_criterion; smoke OK ll −3655.0 (7 min on 1 CPU)
metadata:
  type: project
---
exp3 iter_2/candidate_5_repair_1 (2026-10-31), mixed_sampling_two_cluster_doubt. The base is the mixed_sampling_kind_uncertainty source, with the two-cluster kind patched in. Two-cluster concepts contain every example, so their likelihood is n·member_log_lik(theta, S): only log counts and member shares are precomputed. The first attempt (kind_conflict_shifted_criterion, see [[kind-conflict-shifted-criterion]]) was rejected: RMSE 0.0003 vs beta_share, because the per-target conflict shift collapsed to 0.
**Why:** a graft that combines two non-null components (theta≈0.9, cluster share) is a safer way past the novelty gate than a new parameter that may collapse.
**How to apply:** this machine has nproc=1, so run novelty fits sequentially. Script: /tmp/nov52/nov.py model.py out.npy responses pool. See [[two-cluster-kind-doubt-criterion]], [[mixed-sampling-kind-uncertainty]].
Rebuilt the same day after the slot dir was reset (files reverted, /tmp scripts gone): a python patch script merges mixed_sampling_kind_uncertainty.py with the pair parts of two_cluster_kind_doubt_criterion.py (pair_log_count, cell_pair_member, two_cluster_share). The rebuild passed the smoke test again: ll −3654.6, 4 min. Novelty not rechecked.
