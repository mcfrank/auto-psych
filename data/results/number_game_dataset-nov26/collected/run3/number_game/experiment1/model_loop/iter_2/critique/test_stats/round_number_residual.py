# name: round_number_residual
# description: Among non-example targets, mean over multiple-of-10 targets of (rating minus that set's mean rating over non-example targets); observed below null means people under-endorse round numbers relative to the model (above null means over-endorse).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    d = pd.DataFrame({"s": df["number_set"].astype(str), "r": r, "t": t})[~inset]
    resid = d["r"] - d.groupby("s")["r"].transform("mean")
    m = (d["t"] % 10) == 0
    return float(resid[m].mean()) if m.sum() > 0 else 0.0
