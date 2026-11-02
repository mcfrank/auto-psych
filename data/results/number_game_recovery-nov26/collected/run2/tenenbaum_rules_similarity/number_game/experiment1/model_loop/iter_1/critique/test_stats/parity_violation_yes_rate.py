# name: parity_violation_yes_rate
# description: Among sets whose examples all share one parity, the yes-rate for targets of the opposite parity; observed > null means people respect the shared-parity rule less than the model (model too rule-strict), observed < null means more.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    pmax = np.nanmax(par, axis=1); pmin = np.nanmin(par, axis=1)
    same = pmax == pmin
    tpar = df["target"].astype(int).to_numpy() % 2
    mask = same & (tpar != pmin)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy(float)[mask].mean())
