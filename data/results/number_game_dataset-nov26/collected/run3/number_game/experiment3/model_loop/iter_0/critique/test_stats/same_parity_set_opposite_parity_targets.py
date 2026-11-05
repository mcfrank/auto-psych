# name: same_parity_set_opposite_parity_targets
# description: On 3+ example sets whose examples all have the same parity but do NOT all share a last digit, yes rate for non-example targets of the opposite parity; observed above null means the model's parity-restricted concepts suppress opposite-parity numbers more than people do (observed below = parity under-used).
def test_statistic(df):
    sp = df["number_set"].str.split(",", expand=True)
    v = sp.apply(pd.to_numeric, errors="coerce")
    n = v.notna().sum(axis=1)
    p = v % 2
    d = v % 10
    samepar = p.min(axis=1) == p.max(axis=1)
    samedig = d.min(axis=1) == d.max(axis=1)
    t = df["target"].astype(int)
    sel = (n >= 3) & samepar & (~samedig) & ((t % 2) != p.min(axis=1))
    if sel.sum() == 0:
        return 0.0
    return float(df.loc[sel, "rating"].mean())
