# name: interior_gap_vs_exterior_matched
# description: For sets of 2+ examples, yes rate for targets inside the span but 8-20 from every example minus targets outside the span at the same 8-20 distance; intervals make the interior far more endorsed, so observed below null_mean means people treat interior gaps more like distance-matched exterior targets (similarity-based) than the model, above means a stronger interior fill-in.
def test_statistic(df):
    ex = pd.DataFrame(df["number_set"].astype(str).str.split(",").tolist(), index=df.index).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n_ex = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(y[a].mean() - y[b].mean())
    base = (n_ex >= 2) & (dist >= 8) & (dist <= 20)
    inside = (t > lo) & (t < hi)
    return diff(base & inside, base & ~inside)
