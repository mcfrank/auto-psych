# name: low_alternation_pair_choice_rate
# description: Among trials where both sequences have alternation rate <= 0.5 and the rates differ, the proportion choosing the MORE alternating sequence; observed above the null means people reward extra switching among streaky sequences more strongly than the model's quadratic term implies, below means less.
def test_statistic(df):
    def alt(s):
        return sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: alt(s) for s in u}
    aa = df["sequence_a"].map(m).to_numpy(); ab = df["sequence_b"].map(m).to_numpy()
    sel = (aa <= 0.5) & (ab <= 0.5) & (np.abs(aa - ab) > 1e-9)
    if not sel.any():
        return 0.5
    chose_more = np.where(aa > ab, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    return float(chose_more[sel].mean())
