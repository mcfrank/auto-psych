# name: within_set_response_variance
# description: Mean over number sets (with 10+ trials) of the per-set yes-rate variance p(1-p); observed above null means human responses to a set are more mixed across targets than the model's predictions (less all-or-none), below means more polarized.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) == 0:
        return 0.0
    p = g["mean"].to_numpy(dtype=float)
    return float((p * (1 - p)).mean())
