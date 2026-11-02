# name: endpoint_targets_1_and_100_yes_rate
# description: Yes rate for targets 1 or 100 when they are not among the examples, minus the yes rate for all other non-example targets in the same sets; observed below null means people reject the scale endpoints more than the model predicts (endpoints as non-outputs), above null means they accept them more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values
    t = df["target"].astype(int).values
    isex = (ex == t[:, None]).any(1)
    r = df["rating"].astype(float).values
    end = (t == 1) | (t == 100)
    s = df["number_set"].astype(str).values
    sets_with_end = np.unique(s[end & ~isex])
    insets = np.isin(s, sets_with_end) & ~isex
    a = r[insets & end]; b = r[insets & ~end]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
