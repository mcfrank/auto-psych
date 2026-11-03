# name: within_participant_rate_dispersion
# description: Mean (over the two 2250-row stimulus batches) of the SD of per-participant yes rates, participants found from trial_index resets to 0; observed above null means people differ in overall yes-bias more than the single shared-parameter model produces.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    pid = np.cumsum(ti == 0)
    n = len(df)
    half = np.arange(n) >= n // 2
    out = []
    for h in (False, True):
        m = half == h
        if m.sum() == 0:
            continue
        rates = pd.Series(r[m]).groupby(pid[m]).mean()
        if len(rates) > 1:
            out.append(float(rates.std(ddof=0)))
    return float(np.mean(out)) if out else 0.0
