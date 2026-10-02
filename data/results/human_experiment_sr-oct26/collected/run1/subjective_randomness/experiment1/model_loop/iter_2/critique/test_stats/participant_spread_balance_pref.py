# name: participant_spread_balance_pref
# description: Standard deviation across participants of each participant's proportion choosing the more H/T-balanced sequence among pairs whose imbalance differs by at least 0.25; observed above the null means people differ in how much they weight balance more than the model's single shared balance weight allows.
def test_statistic(df):
    def imb(s):
        return abs(s.count("H") - s.count("T")) / len(s)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    f = {s: imb(s) for s in u}
    ia = df["sequence_a"].map(f).to_numpy(float)
    ib = df["sequence_b"].map(f).to_numpy(float)
    diff = ib - ia
    m = np.abs(diff) >= 0.25 - 1e-9
    if m.sum() == 0:
        return 0.0
    chose_bal = np.where(diff[m] > 0, df["chose_left"].to_numpy(float)[m], 1 - df["chose_left"].to_numpy(float)[m])
    rates = pd.Series(chose_bal).groupby(df["participant_id"].to_numpy()[m]).mean()
    return float(rates.std(ddof=0))
