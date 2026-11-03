# name: far_target_yes_rate
# description: Yes rate to targets more than 30 away from every example in sets of 2+ examples whose examples do not all share parity; observed above null means the model under-predicts diffuse/broad yes responses (e.g. rejects too strongly), below means it over-predicts them via lapse or broad rules.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").reindex(columns=range(4)).values
    ok = ~np.isnan(ex)
    exi = np.where(ok, ex, 0).astype(int)
    t = df["target"].astype(int).values
    n = ok.sum(1)
    par = np.where(ok, exi % 2, -1)
    n_odd = (par == 1).sum(1)
    mixed = (n_odd > 0) & (n_odd < n)
    dist = np.min(np.where(ok, np.abs(exi - t[:, None]), 1000), axis=1)
    m = (n >= 2) & mixed & (dist > 30)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
