# name: common_contained_digit_yes_rate
# description: For sets of 2+ examples that all contain a common digit d but do not all end in the same digit, the yes rate to non-example targets containing d whose distance to the nearest example exceeds 5; observed above null means the model under-weights "contains digit d" rules.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").reindex(columns=range(4)).values
    ok = ~np.isnan(ex)
    exi = np.where(ok, ex, -1000).astype(int)
    n = ok.sum(1)
    sel = np.zeros(len(df), dtype=bool)
    last = np.where(ok, exi % 10, -1)
    same_last = np.all((last == last[:, [0]]) | ~ok, axis=1)
    for d in range(10):
        has = np.zeros_like(ok)
        for c in range(4):
            col = exi[:, c]
            has[:, c] = ok[:, c] & (((col % 10) == d) | (((col // 10) % 10 == d) & (col >= 10)) | ((col == 100) & (d in (0, 1))))
        all_has = np.all(has | ~ok, axis=1) & (n >= 2) & ~same_last
        tc = ((t % 10) == d) | (((t // 10) % 10 == d) & (t >= 10)) | ((t == 100) & (d in (0, 1)))
        sel |= all_has & tc
    dist = np.min(np.where(ok, np.abs(exi - t[:, None]), 1000), axis=1)
    m = sel & (dist > 5)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
