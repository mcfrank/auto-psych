# name: half_balance_choice_rate
# description: Among trials where the two sequences differ in local imbalance (the summed |#H - #T| over the first and second halves, minus the whole-sequence |#H - #T|), the proportion choosing the sequence with GREATER local imbalance; observed below the null means people demand balance within each half (representativeness of sub-sequences) more than the model's global imbalance term predicts, above means less.
def test_statistic(df):
    def loc(s):
        h = len(s) // 2
        f, g = s[:h], s[h:]
        return (abs(f.count("H") - f.count("T")) + abs(g.count("H") - g.count("T"))
                - abs(s.count("H") - s.count("T")))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: loc(s) for s in u}
    la = df["sequence_a"].map(m).to_numpy()
    lb = df["sequence_b"].map(m).to_numpy()
    sel = la != lb
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(la > lb, c, 1 - c)
    return float(chose[sel].mean())
