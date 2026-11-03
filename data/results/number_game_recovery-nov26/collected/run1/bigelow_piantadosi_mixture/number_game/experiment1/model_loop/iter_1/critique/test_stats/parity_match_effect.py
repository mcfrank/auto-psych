# name: parity_match_effect
# description: Among multi-example sets whose examples all share one parity, yes rate for targets of that parity minus yes rate for targets of the other parity; observed above the null means humans use parity more strongly than the model's odd/even rules allow (below means less).
def test_statistic(df):
    sets = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    valid = ~np.isnan(sets)
    n = valid.sum(axis=1)
    par = np.where(valid, np.mod(np.nan_to_num(sets), 2), np.nan)
    n_odd = np.nansum(par, axis=1)
    same = (n >= 2) & ((n_odd == 0) | (n_odd == n))
    set_par = (n_odd > 0).astype(int)
    tpar = df["target"].astype(int).values % 2
    r = df["rating"].astype(float).values
    m1 = same & (tpar == set_par)
    m0 = same & (tpar != set_par)
    if m1.sum() == 0 or m0.sum() == 0:
        return 0.0
    return float(r[m1].mean() - r[m0].mean())
