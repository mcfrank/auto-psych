# name: shared_final_digit_generalisation
# description: Among sets of 2+ examples that all share one final digit d, yes rate on non-example targets ending in d minus yes rate on other targets; observed below null means people generalise the "ends in d" rule less than the model's final-digit discount predicts, above null means more.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    last = np.where(valid, np.nan_to_num(ex) % 10, np.nan)
    first = last[:, 0]
    same = np.where(valid, last == first[:, None], True).all(axis=1)
    t = df["target"].astype(int).to_numpy()
    isex = (ex == t[:, None]).any(axis=1)
    sel = (n >= 2) & same & ~isex
    r = df["rating"].astype(float).to_numpy()
    m = sel & (t % 10 == first)
    o = sel & (t % 10 != first)
    if m.sum() == 0 or o.sum() == 0:
        return 0.0
    return float(r[m].mean() - r[o].mean())
