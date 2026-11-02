# name: inspan_proximity_gap
# description: In sets of >=2 examples, among non-example targets strictly inside [min,max], yes rate for targets within 3 of an example minus targets >=10 from every example; observed above null means people generalize by closeness to individual examples within the span more than the model's interval hypotheses (flat within span) predict.
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
    m = (n >= 2) & ~in_set & (t > lo) & (t < hi)
    return rate(m & (dist <= 3)) - rate(m & (dist >= 10))
