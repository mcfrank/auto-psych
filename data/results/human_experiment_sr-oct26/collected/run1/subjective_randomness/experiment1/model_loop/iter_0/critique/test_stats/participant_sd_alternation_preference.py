# name: participant_sd_alternation_preference
# description: Standard deviation across participants of each participant's rate of choosing the more-alternating sequence (trials where alternation rates differ); observed > null means real individual differences in alternation preference that the single-population model underproduces.
def test_statistic(df):
    def palt(s):
        n = len(s)
        return sum(s[i] != s[i - 1] for i in range(1, n)) / (n - 1) if n > 1 else 0.0
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: palt(s) for s in u}
    pa = df["sequence_a"].map(m).to_numpy()
    pb = df["sequence_b"].map(m).to_numpy()
    y = df["chose_left"].to_numpy()
    mask = np.abs(pa - pb) > 1e-9
    if mask.sum() == 0:
        return 0.0
    chose_more_alt = np.where(pa > pb, y, 1 - y)[mask]
    pid = df["participant_id"].to_numpy()[mask]
    rates = pd.Series(chose_more_alt).groupby(pid).mean()
    return float(rates.std(ddof=0))
