# name: shorter_max_run_by_one_choice_rate
# description: Among trials where the two sequences' longest runs differ by exactly one flip, the proportion choosing the sequence with the SHORTER longest run; observed above the null means a one-flip-longer streak hurts more than the model's linear longest-run share (plus alternation) implies, below means it hurts less.
def test_statistic(df):
    def mr(s):
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: mr(s) for s in u}
    ra = df["sequence_a"].map(m).to_numpy(); rb = df["sequence_b"].map(m).to_numpy()
    sel = np.abs(ra - rb) == 1
    if not sel.any():
        return 0.5
    chose_shorter = np.where(ra < rb, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    return float(chose_shorter[sel].mean())
