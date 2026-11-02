# name: lag1_residual_autocorr
# description: Within-participant correlation between consecutive responses after subtracting each participant's mean yes rate; observed > null means people show response perseveration/streakiness (sequential dependence) that the model, with independent trials, underproduces.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    pid = (ti == 0).cumsum()
    r = df["rating"].to_numpy().astype(float)
    res = r - pd.Series(r).groupby(pid).transform("mean").to_numpy()
    same = pid[1:] == pid[:-1]
    a, b = res[:-1][same], res[1:][same]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
