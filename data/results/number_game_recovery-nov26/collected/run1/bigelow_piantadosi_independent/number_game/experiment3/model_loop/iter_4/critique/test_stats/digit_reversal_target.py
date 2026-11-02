# name: digit_reversal_target
# description: Among non-example two-digit targets more than 3 from every example and sharing no final digit with any example, yes-rate for targets that are the digit reversal of an example (e.g. 43 after 34) minus the yes-rate for the other such targets in the same multi-example sets; observed above the null means people treat digit reversals as similar beyond the model's rules, below means the model over-produces them.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    is_ex = np.any(E == t[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    share_fd = np.any(np.mod(E, 10) == np.mod(t, 10)[:, None], axis=1)
    ti = t.astype(int)
    rev = np.where((ti >= 10) & (ti <= 99), (ti % 10) * 10 + ti // 10, -1).astype(float)
    is_rev = np.any(E == rev[:, None], axis=1) & (rev != t) & (ti % 10 != 0)
    base = (~is_ex) & (n >= 2) & (ti >= 10) & (ti <= 99) & (dist > 3) & (~share_fd)
    hasrev = pd.Series(is_rev & base).groupby(df["number_set"].to_numpy()).transform("any").to_numpy()
    m = base & hasrev
    a, b = m & is_rev, m & (~is_rev)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
