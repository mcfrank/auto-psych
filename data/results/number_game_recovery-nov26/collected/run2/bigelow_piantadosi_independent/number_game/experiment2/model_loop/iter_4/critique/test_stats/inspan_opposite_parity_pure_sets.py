# name: inspan_opposite_parity_pure_sets
# description: In sets of >=2 examples all of one parity, yes rate for non-example opposite-parity targets strictly inside [min,max] of the examples; observed below null means people reject parity-violating in-range targets more than the model (whose ranges endorse them) predicts.
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
    all_odd = ((E % 2 == 1) | pad).all(axis=1)
    all_even = ((E % 2 == 0) | pad).all(axis=1)
    opp = (all_odd & (t % 2 == 0)) | (all_even & (t % 2 == 1))
    return rate((n >= 2) & opp & ~in_set & (t > lo) & (t < hi))
