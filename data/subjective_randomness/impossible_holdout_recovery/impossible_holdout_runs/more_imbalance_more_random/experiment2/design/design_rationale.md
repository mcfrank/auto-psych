# Experiment 2 Design Rationale

**Number of stimuli:** 30
**EIG Range:** 0.2830 – 0.4042

## Discrimination Strategy

The generated design systematically pairs sequences that present stark contrasts across the primary features tracked by the candidate models. Specifically:

1.  **Imbalance vs. Alternation:** The highest EIG pairs frequently contrast a highly alternating sequence (e.g., `"THTH"`, `"HTHTHTHT"`) against a sequence with extreme imbalance (e.g., `"TTTT"`, `"THHHHHHH"`).
2.  **Model Separation:**
    *   This directly pits `inner_loop_model` (which predicts that greater imbalance makes a sequence look *more* random) against models like `high_alternation_rate`, `ideal_alternation_rate`, and `raw_alternation_count` (which predict that more alternations make a sequence look more random).
    *   By testing extremes, we can decisively pull apart whether subjects are tracking head/tail proportions or transition rates.
    *   The variance in lengths within the selected pool (lengths 4 to 8) will also help discriminate length-sensitive models (like raw counts vs. proportions).