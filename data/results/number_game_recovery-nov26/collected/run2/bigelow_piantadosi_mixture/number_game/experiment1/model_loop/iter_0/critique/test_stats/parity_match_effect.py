# name: parity_match_effect
# description: Among sets of >=2 examples that are all odd or all even, yes rate for non-example same-parity targets minus opposite-parity targets; observed above null means people use parity more than the model, below null means the model over-relies on odd/even.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    n = (~np.isnan(ex)).sum(axis=1)
    par = ex % 2
    p0 = par[:, 0]
    same = np.all((par == p0[:, None]) | np.isnan(par), axis=1) & (n >= 2)
    tgt = df["target"].astype(int).to_numpy()
    in_set = (ex == tgt[:, None]).any(axis=1)
    r = df["rating"].astype(float).to_numpy()
    match = same & ~in_set & ((tgt % 2) == p0)
    mis = same & ~in_set & ((tgt % 2) != p0)
    if match.sum() == 0 or mis.sum() == 0:
        return 0.0
    return float(r[match].mean() - r[mis].mean())
