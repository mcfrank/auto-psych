# name: two_example_set_rate_variance
# description: Variance across two-example sets of each set s overall yes-rate; observed above null means sets differ in generalization breadth more than the model predicts (model too homogeneous across sets), below means less.
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
    m = n == 2
    g = pd.Series(r[m]).groupby(df["number_set"].to_numpy()[m]).mean()
    return float(g.var()) if len(g) > 1 else 0.0