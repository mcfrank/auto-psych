---
name: participant-segmentation
description: exp3 responses.csv has no participant_id but rows are ordered; trial_index==0 resets mark 15 participants x 450 trials
metadata:
  type: project
---
In experiment3 `responses.csv`, participants can be recovered with `(trial_index==0).cumsum()`: 15 participants, 450 trials each. Each (set, target) cell has 5 responses. Raw yes-rates come in blocks of 5 (~0.23, ~0.37, ~0.30), so participants seem to share stimulus lists in groups of 5.

**Why:** this makes participant-level PPC statistics possible (breadth dispersion, lag-1 residual dependence) even though there is no id column.
**How to apply:** compute residuals against the cell mean before comparing participants, so differences between stimulus lists don't get counted as individual differences. Related: [[decision-rule-map-screen]].
