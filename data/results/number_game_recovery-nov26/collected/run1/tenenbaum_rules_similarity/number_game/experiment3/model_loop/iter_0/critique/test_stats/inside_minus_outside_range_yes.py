# name: inside_minus_outside_range_yes
# description: Yes rate for non-example targets strictly inside the examples' [min, max] span minus the yes rate for targets outside it, among sets of 2+ examples; observed above null means people treat the example span as a sharper boundary than the (centre-weighted) model predicts, below means people extend past the extremes even more than the model.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    parsed = {s: [int(x) for x in s.split(",")] for s in uniq}
    ex = np.full((len(df), 4), np.nan)
    codes = pd.Categorical(sets, categories=uniq).codes
    table = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = parsed[s][:4]
        table[i, :len(v)] = v
    ex = table[codes]
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    base = (n >= 2) & ~is_ex
    ins = base & (t > lo) & (t < hi)
    out = base & ((t < lo) | (t > hi))
    if not ins.any() or not out.any():
        return 0.0
    return float(y[ins].mean() - y[out].mean())
