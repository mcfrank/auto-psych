---
name: fixed-parity-incumbent-repair
description: Round 2 repair — homogeneity-scaled parity on incumbent was a duplicate; fixed parity dim submitted instead (NLL 1561.3)
metadata:
  type: project
---
Round 2 cell_1 (2026-11-01): candidate_4 (parity attention scaled by how much the examples agree in parity, on variability_tuned_exemplar_attention) was rejected at admission as an exact duplicate of parity_tuned_exemplar_attention. Repair: a fixed parity penalty, the same for every set (`graded_parity_variability_exemplar`), NLL 1561.3. Its MLE p_yes RMSE against the homogeneity-scaled version is 0.004 on the data stimuli and 0.025 on random sets. Other variants on this base, in-sample NLL (base 1574.6): parity-shared "abstract exemplar" floor gives no gain (it makes the fit worse); parity sharpening magnitude attention 1572.3; lapse 1574.6 (no gain).

**Why:** parity-homogeneity attention on the incumbent is already taken twice. **How to apply:** don't propose more homogeneity-scaled parity variants. Related: [[parity-tuned-attention-screen]], [[parity-attention-incumbent-screen]].
