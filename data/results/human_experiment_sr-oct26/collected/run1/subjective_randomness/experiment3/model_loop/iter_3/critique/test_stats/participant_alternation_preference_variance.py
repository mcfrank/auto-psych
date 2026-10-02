# name: participant_alternation_preference_variance
# description: Variance across participants of the rate of choosing the sequence with more switches (pairs whose switch counts differ by at least 2); observed above null_mean means people's alternation preferences are more polarised than the model's personal ideal-rate population yields, below means less.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    m = 8
    def sw(x):
        arr = np.array([list(v.ljust(m, " ")) for v in x])
        return ((arr[:, 1:] != arr[:, :-1]) & (arr[:, 1:] != " ") & (arr[:, :-1] != " ")).sum(1)
    diff = sw(a) - sw(b)
    mask = np.abs(diff) >= 2
    cl = df["chose_left"].to_numpy().astype(float)
    chose_more = np.where(diff > 0, cl, 1 - cl)[mask]
    pid = df["participant_id"].to_numpy()[mask]
    g = pd.Series(chose_more).groupby(pid).agg(["mean", "size"])
    g = g[g["size"] >= 5]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].var())
