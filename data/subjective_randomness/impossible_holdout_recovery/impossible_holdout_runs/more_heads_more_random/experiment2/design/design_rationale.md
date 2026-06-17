# Experiment Design Rationale

**Number of stimuli:** 20
**Expected Information Gain (EIG) range:** 0.2661 – 0.4558

**Methodology:**
A tractable pool of 300 candidate sequence pairs was randomly generated, featuring both equal-length and unequal-length pairs (with sequence lengths bounded between 2 and 8, as per the experimental constraints). 

**Discrimination Strategy:**
The computed EIG algorithm prioritized pairs that robustly split the predictive distributions of the available cognitive models. Noticeably, many of the top-ranked stimuli pit highly skewed or minimally alternating sequences (e.g., `HHHHHHH`, `HHHHHHHH`) against sequences with rich internal structure (e.g., `HHTTHTT`, `HTHTHTTH`). 

This structure actively tests models sensitive to extremes like `squared_heads_heuristic` or `length_scaled_head_difference` against those indexing patterns and alternations, such as `encoding_compressibility` or `bayesian_diagnosticity`, thereby maximizing the diagnosticity of each forced-choice trial.
