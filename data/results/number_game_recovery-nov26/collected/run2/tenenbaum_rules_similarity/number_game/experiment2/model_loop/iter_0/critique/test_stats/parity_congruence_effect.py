# name: parity_congruence_effect
# description: Among sets (n>=2) whose examples all share parity, yes rate for non-example targets of the same parity minus those of opposite parity; observed above the null means people use parity more than the model's weighted odd/even rules allow (below = less).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    par = np.where(valid, np.mod(np.nan_to_num(ex), 2), np.nan)
    allodd = (np.nansum(par, axis=1) == n)
    alleven = (np.nansum(par, axis=1) == 0)
    t = df["target"].astype(int).values
    notin = ~(ex == t[:, None].astype(float)).any(axis=1)
    same = (allodd & (t % 2 == 1)) | (alleven & (t % 2 == 0))
    base = (n >= 2) & (allodd | alleven) & notin
    r = df["rating"].values.astype(float)
    a = base & same; b = base & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
