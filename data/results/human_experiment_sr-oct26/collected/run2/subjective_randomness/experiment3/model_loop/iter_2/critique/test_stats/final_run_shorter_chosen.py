# name: final_run_shorter_chosen
# description: Among pairs whose final runs (trailing streak) differ in length, the rate of choosing the sequence with the shorter final run; observed above null means a recency-weighted aversion to streaks at the end that the position-blind model under-produces, below means the model overstates it.
def test_statistic(df):
    def fr(col):
        s = df[col]
        last = s.str[-1]
        stripped_h = s.str.rstrip("H").str.len()
        stripped_t = s.str.rstrip("T").str.len()
        return (s.str.len() - np.where(last == "H", stripped_h, stripped_t)).to_numpy()
    fa = fr("sequence_a"); fb = fr("sequence_b")
    m = fa != fb
    if not m.any():
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.where(fa[m] < fb[m], c, 1 - c).mean())
