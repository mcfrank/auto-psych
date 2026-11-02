# name: parity_match_effect
# description: For multi-example sets whose examples all share one parity, yes-rate for same-parity targets minus opposite-parity targets; observed > null means people rely on parity more strongly than the model predicts (observed < null: less).
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    allodd = np.nansum(par == 1, axis=1) == n
    alleven = np.nansum(par == 0, axis=1) == n
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = (n >= 2) & (allodd | alleven)
    setpar = np.where(allodd, 1, 0)
    same = sel & ((t % 2) == setpar)
    opp = sel & ((t % 2) != setpar)
    if same.sum() == 0 or opp.sum() == 0:
        return 0.0
    return float(r[same].mean() - r[opp].mean())
