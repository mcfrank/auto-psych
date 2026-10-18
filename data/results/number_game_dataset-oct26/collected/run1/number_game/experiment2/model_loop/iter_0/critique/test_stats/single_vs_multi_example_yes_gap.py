# name: single_vs_multi_example_yes_gap
# description: Overall yes-rate on single-example sets minus yes-rate on 3-4-example sets (targets not in set); observed differing from model mean means the model mis-captures how generalization breadth changes with number of examples (size principle).
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    inset = (X == t[:, None]).any(axis=1)
    a = (n == 1) & ~inset; b = (n >= 3) & ~inset
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
