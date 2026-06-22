import numpy as np
import pandas as pd


# name: short_sequence_imbalance_penalty
# description: The correlation between sequence A's relative imbalance advantage and the choice rate, restricted to trials where both sequences are short (n <= 5), testing if the model's small-sample smoothing correctly captures human behavior.
def test_statistic(df):
    mask = (df["n_a"] <= 5) & (df["n_b"] <= 5)
    if mask.sum() < 2:
        return np.nan

    # Advantage for A in terms of imbalance (positive means A is closer to 0.5)
    imb_advantage_a = df.loc[mask, "imbalance_b"] - df.loc[mask, "imbalance_a"]
    chose_left = df.loc[mask, "chose_left"]

    # Return the Pearson correlation
    # Add small noise to avoid division by zero if all values are identical
    corr = np.corrcoef(imb_advantage_a, chose_left)[0, 1]
    return corr if not np.isnan(corr) else 0.0
