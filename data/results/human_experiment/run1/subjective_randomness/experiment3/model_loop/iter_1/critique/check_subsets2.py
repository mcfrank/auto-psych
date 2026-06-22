import pandas as pd
import numpy as np

df = pd.read_csv(
    "/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/responses.csv"
)

mask = (df["imbalance_a"] > 0.4) & (df["imbalance_b"] > 0.4) & (df["n_a"] != df["n_b"])
print("Length penalty scaling mask sum:", mask.sum())
