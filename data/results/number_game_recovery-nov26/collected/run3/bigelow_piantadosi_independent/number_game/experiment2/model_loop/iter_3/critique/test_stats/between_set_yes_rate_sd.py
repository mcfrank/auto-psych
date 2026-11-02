# name: between_set_yes_rate_sd
# description: Standard deviation across sets (with 20+ non-example trials) of the per-set yes rate on non-example targets; observed above null means sets differ in overall generalisation more than the model predicts (model too homogeneous), below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    d = pd.DataFrame({"s": df["number_set"].astype(str).to_numpy(), "y": y})[~is_ex]
    g = d.groupby("s")["y"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
