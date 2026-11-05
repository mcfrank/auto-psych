# name: unstructured_set_target_dispersion
# description: For multi-example sets sharing neither parity nor final digit, the response-weighted mean over sets of the across-target SD of pair-level yes-rates (pairs with >=5 responses); observed above null means people discriminate between targets of weakly-structured sets more sharply than the model's near-flat overlap predicts, below means the model is too target-specific.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    valid = ~np.isnan(E)
    P = np.mod(E, 2)
    samep = (np.sum((P == 1) & valid, axis=1) == n) | (np.sum((P == 0) & valid, axis=1) == n)
    D = np.mod(E, 10)
    samed = np.nanmin(D, axis=1) == np.nanmax(D, axis=1)
    m = (n >= 2) & (~samep) & (~samed)
    sub = df.loc[m, ["number_set", "target", "rating"]]
    if len(sub) == 0:
        return 0.0
    g = sub.groupby(["number_set", "target"])["rating"].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= 5]
    if len(g) == 0:
        return 0.0
    s = g.groupby("number_set").agg(sd=("mean", lambda x: float(np.std(x))), k=("mean", "size"), w=("size", "sum"))
    s = s[s["k"] >= 3]
    if len(s) == 0:
        return 0.0
    return float(np.average(s["sd"], weights=s["w"]))
