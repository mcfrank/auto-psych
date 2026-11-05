# name: even_set_parity_contrast
# description: For sets whose examples are all even, yes rate for even targets minus odd targets (non-example targets); observed above null means people rely on parity more than the model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    alleven = ((ex % 2 == 0) | ex.isna()).all(axis=1)
    return m(alleven & (~inset) & (t % 2 == 0)) - m(alleven & (~inset) & (t % 2 == 1))
