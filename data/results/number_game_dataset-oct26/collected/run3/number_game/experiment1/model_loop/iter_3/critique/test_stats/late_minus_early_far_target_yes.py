# name: late_minus_early_far_target_yes
# description: Yes-rate on far targets (>20 from every example) with trial_index above the median minus at/below it; observed above null means blind/loose "yes" to concept-excluded targets grows over the session, which a constant lapse rate cannot produce.
def _feat(df):
    sets = {s: [int(x) for x in str(s).split(",")] for s in df["number_set"].unique()}
    ex = df["number_set"].map(sets)
    t = df["target"].astype(int).to_numpy()
    lo = df["number_set"].map({s: min(v) for s, v in sets.items()}).to_numpy()
    hi = df["number_set"].map({s: max(v) for s, v in sets.items()}).to_numpy()
    n = df["number_set"].map({s: len(v) for s, v in sets.items()}).to_numpy()
    arr = np.full((len(df), 4), np.nan)
    for s, v in sets.items():
        m = (df["number_set"] == s).to_numpy()
        arr[m, :len(v)] = v
    dist = np.nanmin(np.abs(arr - t[:, None]), axis=1)
    near = arr[np.arange(len(df)), np.nanargmin(np.abs(arr - t[:, None]), axis=1)]
    return sets, t, lo, hi, n, arr, dist, near, df["rating"].to_numpy().astype(float)


def _m(x):
    return float(np.mean(x)) if len(x) else 0.0


def test_statistic(df):
    sets, t, lo, hi, n, arr, dist, near, r = _feat(df)
    ti = df["trial_index"].to_numpy()
    med = np.median(ti)
    far = dist > 20
    return _m(r[far & (ti > med)]) - _m(r[far & (ti <= med)])