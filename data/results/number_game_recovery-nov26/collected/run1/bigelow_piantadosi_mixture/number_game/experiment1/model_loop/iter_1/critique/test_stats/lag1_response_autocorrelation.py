# name: lag1_response_autocorrelation
# description: Within-participant lag-1 correlation between consecutive yes/no responses (rows in session order, participants split where trial_index resets); observed above the null means responses are sequentially dependent (perseveration/streaks) beyond what stimulus-driven independent responding predicts.
def test_statistic(df):
    t = df["trial_index"].astype(int).values
    r = df["rating"].astype(float).values
    same = np.diff(t) == 1
    a, b = r[:-1][same], r[1:][same]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
