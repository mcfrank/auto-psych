# name: double_half_target_contrast
# description: Set-demeaned yes rate for targets equal to twice or half of a shown example, in sets that are not all powers of 2 (target not shown); observed above null means people generalize by doubling/halving relations the model lacks, below means the model over-credits them.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].to_numpy(dtype=float)
    valid = ~np.isnan(ex)
    e = np.nan_to_num(ex).astype(int)
    p2 = np.isin(e, [1, 2, 4, 8, 16, 32, 64])
    all_p2 = (p2 | ~valid).all(axis=1)
    rel = valid & ((2 * e == t[:, None]) | ((e % 2 == 0) & (e // 2 == t[:, None])))
    shown = (valid & (e == t[:, None])).any(axis=1)
    flag = rel.any(axis=1) & ~shown & ~all_p2
    g = pd.DataFrame({"set": df["number_set"].to_numpy(), "y": y})
    tot = g.groupby("set")["y"].transform("sum").to_numpy()
    cnt = g.groupby("set")["y"].transform("count").to_numpy()
    other = (tot - y) / np.maximum(cnt - 1, 1)
    if flag.sum() == 0:
        return 0.0
    return float(np.mean(y[flag] - other[flag]))
