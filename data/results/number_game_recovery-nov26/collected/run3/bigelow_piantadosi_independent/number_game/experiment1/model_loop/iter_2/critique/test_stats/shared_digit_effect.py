# name: shared_digit_effect
# description: Yes rate for non-example targets containing a digit that appears in every example minus yes rate for non-example targets containing none of the examples' digits (multi-example sets with a common digit); observed > null means people generalize by shared digits more than the model, observed < null means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int)
    r = df["rating"].astype(float).values
    ex = s.str.split(",", expand=True)
    n_ex = s.str.count(",") + 1
    ts = t.astype(str)
    common = np.ones(len(df), dtype=bool)
    tgt_common = np.zeros(len(df), dtype=bool)
    tgt_any = np.zeros(len(df), dtype=bool)
    has_common = np.zeros(len(df), dtype=bool)
    is_ex = np.zeros(len(df), dtype=bool)
    for c in ex.columns:
        is_ex |= (ex[c].fillna("-1").astype(int) == t).values
    for d in "0123456789":
        in_all = np.ones(len(df), dtype=bool)
        in_any = np.zeros(len(df), dtype=bool)
        for c in ex.columns:
            col = ex[c]
            present = col.notna().values
            contains = col.fillna("").str.contains(d).values
            in_all &= (~present) | contains
            in_any |= contains
        td = ts.str.contains(d).values
        has_common |= in_all
        tgt_common |= in_all & td
        tgt_any |= in_any & td
    base = (n_ex.values >= 2) & has_common & ~is_ex
    a = base & tgt_common
    b = base & ~tgt_any
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
