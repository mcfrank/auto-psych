# name: equal_switch_pair_consensus
# description: Mean pair-level consensus |P(choose one sequence) - 0.5| over pairs whose two sequences have the same number of switches (side-free pair identity); observed above null_mean means people agree on preferences driven by something other than alternation that the model's secondary features under-predict, below means the model's secondary features make such pairs too decisive.
def test_statistic(df):
    a = df["sequence_a"].to_numpy().astype(str); b = df["sequence_b"].to_numpy().astype(str)
    def sw(x):
        arr = np.array([list(v.ljust(8, " ")) for v in x])
        return ((arr[:, 1:] != arr[:, :-1]) & (arr[:, 1:] != " ") & (arr[:, :-1] != " ")).sum(1)
    mask = sw(a) == sw(b)
    if mask.sum() == 0:
        return 0.0
    s = a < b
    key = np.char.add(np.char.add(np.where(s, a, b), "|"), np.where(s, b, a))[mask]
    cl = df["chose_left"].to_numpy().astype(float)
    cf = np.where(s, cl, 1 - cl)[mask]
    r = pd.Series(cf).groupby(key).mean().to_numpy()
    return float(np.abs(r - 0.5).mean())
