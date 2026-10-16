# name: ends_with_switch_preference
# description: Among trials where exactly one sequence ends with a switch (last two flips differ), the proportion choosing that sequence; observed > null means people weight the ending (recency / gambler's-fallacy) more than the position-invariant model predicts, < null the reverse.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    ea = (a.str[-1] != a.str[-2]).to_numpy()
    eb = (b.str[-1] != b.str[-2]).to_numpy()
    m = ea != eb
    y = df["chose_left"].to_numpy()
    return float(np.where(ea, y, 1 - y)[m].mean())
