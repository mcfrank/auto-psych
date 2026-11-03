# name: participant_parity_discrimination_sd
# description: SD across participants of (yes rate on parity-congruent targets minus yes rate on parity-incongruent targets) for sets of >=2 examples that all share a parity; observed above null means the model under-produces individual differences in how sharply people generalise by rule.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    n = (~np.isnan(ex)).sum(axis=1)
    odd = np.nansum(ex % 2 == 1, axis=1)
    allodd = odd == n
    alleven = odd == 0
    t = df["target"].astype(int).to_numpy()
    sel = (n >= 2) & (allodd | alleven)
    cong = np.where(allodd, t % 2 == 1, t % 2 == 0)
    r = df["rating"].to_numpy().astype(float)
    d = pd.DataFrame({"p": pid[sel], "c": cong[sel], "r": r[sel]})
    m = d.groupby(["p", "c"])["r"].mean().unstack()
    diff = (m.get(True) - m.get(False)).dropna()
    if len(diff) < 2:
        return 0.0
    return float(diff.std(ddof=0))
