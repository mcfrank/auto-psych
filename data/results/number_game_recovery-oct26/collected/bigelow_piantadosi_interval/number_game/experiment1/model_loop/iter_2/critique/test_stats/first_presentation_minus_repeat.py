# name: first_presentation_minus_repeat
# description: Within each participant session (trial_index resets to 0), yes rate on the first trial a given example set appears minus yes rate on later trials with that same set; observed differing from null means generalization shifts with repeated exposure to a set (positive = narrowing, negative = broadening), which the stationary model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy().astype(int)
    pid = np.cumsum(ti == 0)
    key = pd.Series(pid.astype(str)) + "|" + pd.Series(df["number_set"].astype(str).to_numpy())
    order = np.lexsort((ti, pid))
    first = np.zeros(len(ti), dtype=bool)
    first[order] = ~key.iloc[order].duplicated().to_numpy()
    r = df["rating"].astype(float).to_numpy()
    rep = ~first
    if not first.any() or not rep.any():
        return 0.0
    return float(r[first].mean() - r[rep].mean())
