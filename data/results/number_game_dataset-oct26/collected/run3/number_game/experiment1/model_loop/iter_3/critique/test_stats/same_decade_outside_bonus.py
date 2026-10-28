# name: same_decade_outside_bonus
# description: Among outside-range targets at distance 3-15 from the nearest example, yes-rate when the target shares its tens decade with some example minus when it does not; observed above null means people group numbers by decade more than the model predicts.
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
    dec = np.floor(np.minimum(arr, 99) / 10)
    same = np.any(dec == np.floor(np.minimum(t, 99) / 10)[:, None], axis=1)
    ok = ((t < lo) | (t > hi)) & (dist >= 3) & (dist <= 15)
    return _m(r[ok & same]) - _m(r[ok & ~same])