# name: consensus_yes_cell_fraction
# description: Among distinct (set, target) cells with >= 5 responses whose target is not an example, the fraction with yes-rate >= 0.85; observed > null means people agree on generalizing to some non-example targets more uniformly than the model's guessing/stray noise allows (observed < null: the model is over-confident).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    inset = np.any(E == t[:, None], axis=1)
    d = df.loc[~inset]
    key = d["number_set"].astype(str) + "|" + d["target"].astype(str)
    g = d["rating"].astype(float).groupby(key).agg(["mean", "size"])
    g = g[g["size"] >= 5]
    return float((g["mean"] >= 0.85).mean()) if len(g) else 0.0
