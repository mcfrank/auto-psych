# name: participant_consensus_agreement_sd
# description: SD across participants (sessions delimited by trial_index resets to 0) of each participant's mean agreement 1-|r - leave-one-out cell mean| with the others' responses in the same (set, target) cell; observed above null means participants differ in how noisy/rule-consistent they are (individual differences the single-parameter model lacks), below means more homogeneous.
def test_statistic(df):
    ti = df["trial_index"].to_numpy().astype(int)
    pid = np.cumsum(ti == 0)
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    r = df["rating"].astype(float).to_numpy()
    s = pd.Series(r).groupby(key.to_numpy()).transform("sum").to_numpy()
    n = pd.Series(r).groupby(key.to_numpy()).transform("size").to_numpy()
    ok = n >= 2
    if not ok.any():
        return 0.0
    loo = (s[ok] - r[ok]) / (n[ok] - 1)
    agree = 1 - np.abs(r[ok] - loo)
    per = pd.Series(agree).groupby(pid[ok]).mean()
    if len(per) < 2:
        return 0.0
    return float(per.std(ddof=1))
