# name: pair_out_of_span_near_yes
# description: In 2-example sets, yes rate for non-example targets outside [min,max] but within 5 of the nearer edge; observed below null means people stop more sharply at the span of two examples than the model predicts (model over-extends ranges beyond pairs).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    return rate((n == 2) & ~in_set & ((t < lo) | (t > hi)) & (dist <= 5))
