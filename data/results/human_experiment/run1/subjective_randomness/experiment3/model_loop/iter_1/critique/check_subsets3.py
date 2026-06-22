import pandas as pd
import numpy as np

df = pd.read_csv(
    "/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/responses.csv"
)

mask = (
    (df["p_alts_a"] < 0.4)
    & (df["p_alts_b"] > 0.6)
    & (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.2)
)
print("Under vs Over alternation mask sum:", mask.sum())
