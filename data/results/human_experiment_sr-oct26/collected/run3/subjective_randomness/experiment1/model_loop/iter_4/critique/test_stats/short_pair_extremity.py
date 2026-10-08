# name: short_pair_extremity
# description: Mean |pair choice rate - 0.5| over pairs of length <=5 only; observed above null_mean means people are still more decisive on short sequences than the length-normalised model predicts, below means the model now over-predicts short-pair decisiveness.
def test_statistic(df):
    a = df["sequence_a"].astype(str); b = df["sequence_b"].astype(str)
    m = (a.str.len() <= 5).to_numpy()
    if m.sum() == 0:
        return 0.0
    lo = np.where(a < b, a, b); key = lo + "|" + np.where(a < b, b, a)
    chose_lo = np.where(a.to_numpy() == lo, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy()).astype(float)
    r = pd.Series(chose_lo[m]).groupby(key[m]).mean()
    return float(np.mean(np.abs(r - 0.5)))
