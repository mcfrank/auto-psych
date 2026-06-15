# Theory Report — Experiment 3

## run_length_prototype

**Hypothesis:** People judge sequences by how close the maximum run length is to an internal prototype for a random sequence — both too-long streaks (obvious repetition) and too-short maximum runs (forced alternation, where every flip changes) look non-random, so the judgment follows a symmetric Gaussian penalty around an ideal normalized run length.

**Motivation:** The winning model from experiment 2 (`inner_loop_model`, 91.9% posterior mass) uses alternation proportion and balance as its two cues, completely ignoring run structure. The only run-based model in the existing set (`iter0_candidate1`, from the experiment 2 inner loop) tested whether *shorter* max runs always look more random — a monotone hypothesis — and received negligible posterior mass (≈0.004). But the monotone assumption may be wrong: a sequence where every flip changes (max_run = 1, perfectly alternating) does not look random; it looks patterned. A prototype model for max run length captures this symmetric intuition and is a genuinely distinct hypothesis: it uses *only* run structure, and it predicts that the ideal max run is intermediate (around 1/3 of sequence length), not minimal.

**Mechanism:** The score for each sequence is `−β × (max_run_norm − θ_run)²`, where `max_run_norm = max_run / n` is the normalized max run and `θ_run` is a learned ideal (Beta(2,3) prior, mode ≈ 0.33). The preference is determined by which sequence's normalized max run is closer to the prototype. This is distinct from all existing models: it uses a single run-structure cue rather than alternation/balance, it is symmetric around an internal ideal rather than monotone, and it does not combine run length with any other feature.

---

## length_sensitive_2d_prototype

**Hypothesis:** People evaluate sequences using the same two-dimensional prototype as the winning account (alternation rate + balance, Gaussian decay), but their confidence in the deviation signal scales linearly with sequence length — a longer sequence that deviates from the prototype provides proportionally stronger evidence of non-randomness than a shorter sequence with the same proportional deviation.

**Motivation:** The winning model (`inner_loop_model`) treats a length-4 and a length-8 sequence with the same `p_alts` deviation identically. But statistical reasoning suggests that the same proportion deviation is more *diagnostic* when the sequence is longer (more coin flips = stronger evidence). The experiment pairs sequences of different lengths (e.g., 4 vs 6, 6 vs 8), which is exactly the condition where length-dependent weighting would produce different predictions from the proportion-scale model. The catastrophically bad `length_sensitive_alternation` model (ELPD = −12,323 in experiment 2) tested a count-scale (n² scaling, one-dimensional) alternative; this model instead tests linear scaling (n scaling) in two dimensions — a fundamentally different functional form that avoids the quadratic inflation that destroyed the count-scale model.

**Mechanism:** The score for each sequence is `−(n/4) × [β_alt × (p_alts − θ_alt)² + β_bal × (p − 0.5)²]`, where dividing by 4 normalizes the weight so that the minimum-length sequence (n=4) has a unit multiplier and the parameters `β_alt`, `β_bal` remain on a comparable scale to the winning model. When comparing sequences of different lengths, this model predicts that the longer sequence's prototype distance matters proportionally more, which is the key empirical distinction from the unweighted winning model.
