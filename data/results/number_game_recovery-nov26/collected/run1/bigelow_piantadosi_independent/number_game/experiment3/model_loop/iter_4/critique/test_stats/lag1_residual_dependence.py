# name: lag1_residual_dependence
# description: Within-participant correlation between consecutive trials' residual responses (rating minus cell yes-rate minus the participant's mean residual); observed above the null means responses perseverate (a yes makes the next yes more likely) beyond what the trial-independent model produces, below means people alternate.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = (ti == 0).cumsum()
    r = df["rating"].to_numpy().astype(float)
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    res = r - pd.Series(r).groupby(key.to_numpy()).transform("mean").to_numpy()
    res = res - pd.Series(res).groupby(pid).transform("mean").to_numpy()
    ok = (pid[1:] == pid[:-1]) & (np.diff(ti) == 1)
    a, b = res[1:][ok], res[:-1][ok]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
