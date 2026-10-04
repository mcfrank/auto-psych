# name: start_symbol_heads_preference
# description: Among pairs where one sequence starts with H and the other with T, the rate of choosing the H-starting sequence; observed above null means the first flip's label matters beyond overall head count (model under-produces this), below means the opposite.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    sa, sb = a.str[0], b.str[0]
    mask = sa != sb
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"][mask]
    return float(np.where((sa == "H")[mask], c, 1 - c).mean())
