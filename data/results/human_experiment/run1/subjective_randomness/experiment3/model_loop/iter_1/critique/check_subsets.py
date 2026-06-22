import pandas as pd
import numpy as np

df = pd.read_csv(
    "/scratch/users/benpry/auto-psych/outer_loop_live/run1/data/subjective_randomness/experiment3/model_loop/responses.csv"
)

print("Total trials:", len(df))

# 1. Extreme proportion deviation
mask1 = (df["imbalance_a"] > 0.6) & (df["imbalance_b"] < 0.2)
print("1. Extreme proportion deviation:", mask1.sum())

# 2. Extreme alternation rate (zero or one) vs normal
mask2 = (
    ((df["p_alts_a"] == 0) | (df["p_alts_a"] == 1))
    & (df["p_alts_b"] > 0.2)
    & (df["p_alts_b"] < 0.8)
)
print("2. Extreme alternation rate vs normal:", mask2.sum())

# 3. Max run length controlling for p and p_alts
mask3 = (
    (np.abs(df["p_a"] - df["p_b"]) < 0.2)
    & (np.abs(df["p_alts_a"] - df["p_alts_b"]) < 0.2)
    & (df["max_run_a"] > df["max_run_b"] + 1)
)
print("3. Max run length difference:", mask3.sum())

# 4. Periodicity difference controlling for p and p_alts
mask4 = (
    (np.abs(df["p_a"] - df["p_b"]) < 0.2)
    & (np.abs(df["p_alts_a"] - df["p_alts_b"]) < 0.2)
    & (df["periodicity_a"] > df["periodicity_b"] + 0.2)
)
print("4. Periodicity difference:", mask4.sum())

# 5. Length effect preference
mask5 = df["n_a"] > df["n_b"]
print("5. Length effect (n_a > n_b):", mask5.sum())

# 6. Asymmetry Heads vs Tails
mask6 = (
    (df["p_a"] > 0.6)
    & (df["p_b"] < 0.4)
    & (np.abs((df["p_a"] - 0.5) + (df["p_b"] - 0.5)) < 0.1)
)
print("6. Asymmetry Heads vs Tails:", mask6.sum())

# 7. Short vs Long sequences overall choice
mask7 = (df["n_a"] <= 4) & (df["n_b"] >= 7)
print("7. Short vs Long sequences:", mask7.sum())

# 8. Under-alternation vs Over-alternation
mask8 = (
    (df["p_alts_a"] <= 0.3)
    & (df["p_alts_b"] >= 0.7)
    & (np.abs(df["imbalance_a"] - df["imbalance_b"]) < 0.2)
)
print("8. Under vs Over alternation:", mask8.sum())
