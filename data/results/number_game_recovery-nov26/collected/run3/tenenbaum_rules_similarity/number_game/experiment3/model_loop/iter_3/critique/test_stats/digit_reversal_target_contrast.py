# name: digit_reversal_target_contrast
# description: Yes rate for targets that are the digit reversal of a shown two-digit example (e.g. 38 for 83; target not itself shown) minus the mean yes rate of that set's other targets; observed below null means people reject reversals more than the model's shared-digit rules predict, above means they treat reversals as same-concept.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    e = np.nan_to_num(ex).astype(int)
    tens, units = e // 10, e % 10
    revable = valid & (e >= 10) & (e <= 99) & (units != 0) & (tens != units)
    rev = units * 10 + tens
    is_rev = (revable & (rev == t[:, None])).any(axis=1)
    shown = (valid & (e == t[:, None])).any(axis=1)
    is_rev &= ~shown
    sets = df["number_set"].to_numpy()
    g = pd.DataFrame({"set": sets, "y": y, "r": is_rev})
    tot = g.groupby("set")["y"].transform("sum")
    cnt = g.groupby("set")["y"].transform("count")
    other_mean = (tot - g["y"]) / np.maximum(cnt - 1, 1)
    if is_rev.sum() == 0:
        return 0.0
    return float(np.mean(y[is_rev] - other_mean.to_numpy()[is_rev]))
