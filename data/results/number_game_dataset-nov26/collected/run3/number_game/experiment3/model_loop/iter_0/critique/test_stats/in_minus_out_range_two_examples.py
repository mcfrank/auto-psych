# name: in_minus_out_range_two_examples
# description: On 2-example sets, yes rate for non-example targets strictly inside the examples' range minus yes rate for targets outside it; observed above null means the model under-weights range/interval generalization relative to rules for pairs.
def test_statistic(df):
    sp = df["number_set"].str.split(",", expand=True)
    n = sp.notna().sum(axis=1)
    v = sp.apply(pd.to_numeric, errors="coerce")
    lo, hi = v.min(axis=1), v.max(axis=1)
    t = df["target"].astype(int)
    isex = v.eq(t, axis=0).any(axis=1)
    base = (n == 2) & (~isex)
    inn = base & (t > lo) & (t < hi)
    out = base & ((t < lo) | (t > hi))
    if inn.sum() == 0 or out.sum() == 0:
        return 0.0
    r = df["rating"]
    return float(r[inn].mean() - r[out].mean())
