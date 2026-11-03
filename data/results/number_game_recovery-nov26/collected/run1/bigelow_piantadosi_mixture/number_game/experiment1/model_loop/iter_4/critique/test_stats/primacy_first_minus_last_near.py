# name: primacy_first_minus_last_near
# description: In multi-example sets, yes-rate for non-example targets within 6 of the first-shown example (and >6 from all others) minus that for targets within 6 of the last-shown example only; observed > null means people weight the first example more than the sticky-first-guess model, observed < null means the model overstates primacy.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float).values
    n = parts.notna().sum(axis=1).values
    P = parts.values
    first = P[:, 0]
    last = P[np.arange(len(P)), n - 1]
    dist = np.abs(P - t[:, None])
    isex = np.nansum(dist == 0, axis=1) > 0
    near = np.where(np.isnan(dist), False, dist <= 6)
    cnt = near.sum(axis=1)
    near_first = (np.abs(first - t) <= 6) & (cnt == 1)
    near_last = (np.abs(last - t) <= 6) & (cnt == 1)
    ok = (n >= 2) & ~isex
    y = df["rating"].values.astype(float)
    a, b = ok & near_first, ok & near_last
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
