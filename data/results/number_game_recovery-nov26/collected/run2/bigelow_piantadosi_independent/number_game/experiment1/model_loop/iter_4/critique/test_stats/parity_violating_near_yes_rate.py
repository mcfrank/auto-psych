# name: parity_violating_near_yes_rate
# description: Yes rate for non-example targets of the OPPOSITE parity to all examples, lying within 3 of some example, among sets of >=2 examples that are all even or all odd; t_observed above null_mean means people generalize by numerical proximity despite a parity violation more than the model (whose parity rule swamps intervals) predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    allsame = (np.nanmin(par, axis=1) == np.nanmax(par, axis=1)) & (n >= 2)
    setpar = np.nanmax(par, axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = allsame & ((t % 2) != setpar) & (d > 0) & (d <= 3)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
