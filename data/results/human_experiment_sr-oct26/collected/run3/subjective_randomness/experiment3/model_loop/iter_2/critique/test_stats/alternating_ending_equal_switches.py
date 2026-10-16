# name: alternating_ending_equal_switches
# description: Among pairs with equal switch counts where exactly one sequence's last two flips differ, the rate of choosing the sequence ending in a switch; observed above null means people find a sequence ending in alternation (vs ending in a repeat) more random than the position-blind model predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    sa = (a.str.count("HT") + a.str.count("TH")).values
    sb = (b.str.count("HT") + b.str.count("TH")).values
    ea = (a.str[-1] != a.str[-2]).values; eb = (b.str[-1] != b.str[-2]).values
    y = df["chose_left"].values
    m = (sa == sb) & (ea != eb)
    if not m.any():
        return 0.5
    return float(np.where(ea, y, 1 - y)[m].mean())
