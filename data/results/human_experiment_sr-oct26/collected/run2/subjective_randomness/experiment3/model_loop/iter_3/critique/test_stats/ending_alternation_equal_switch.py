# name: ending_alternation_equal_switch
# description: Among pairs with equal numbers of switches where exactly one sequence ends with a switch (last two flips differ), the rate of choosing the sequence that ends with a switch; observed above null means people weigh the ending (recency, avoiding a terminal repeat) more than the model, which is blind to switch position, below means the reverse.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    seqs = pd.unique(pd.concat([a, b]))
    nsw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    ea = (a.str[-1] != a.str[-2]).astype(int)
    eb = (b.str[-1] != b.str[-2]).astype(int)
    m = (a.map(nsw) == b.map(nsw)) & (ea != eb) & (a.str.len() >= 3)
    if m.sum() == 0:
        return 0.5
    ch = np.where(ea[m] == 1, df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(ch))
