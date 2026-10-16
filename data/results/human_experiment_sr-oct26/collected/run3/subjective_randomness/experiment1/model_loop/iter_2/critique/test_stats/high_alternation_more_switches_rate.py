# name: high_alternation_more_switches_rate
# description: Among pairs where both sequences switch on at least half their transitions and switch counts differ, the proportion of choices for the sequence with MORE switches; observed below null_mean means people stop rewarding extra alternation (or penalise it) in the high-switch regime more than the model's monotone over-alternation term predicts, above means they reward it more.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def sw(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    u = pd.unique(pd.concat([a, b]))
    m = {s: sw(s) for s in u}
    swa = a.map(m).to_numpy(); swb = b.map(m).to_numpy()
    t = a.str.len().to_numpy() - 1
    sel = (swa >= t / 2) & (swb >= t / 2) & (swa != swb)
    if sel.sum() == 0:
        return 0.5
    cl = df["chose_left"].to_numpy()[sel]
    c = np.where(swa[sel] > swb[sel], cl, 1 - cl)
    return float(np.mean(c))
