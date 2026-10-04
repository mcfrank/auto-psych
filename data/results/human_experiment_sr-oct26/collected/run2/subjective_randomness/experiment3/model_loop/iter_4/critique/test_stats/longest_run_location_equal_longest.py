# name: longest_run_location_equal_longest
# description: Among pairs with equal switch counts and equal longest-run length where exactly one sequence has its longest run touching an edge (first or last flip), the rate of choosing the sequence whose longest run is interior; observed above null means people penalise edge streaks (start or end) more than the model predicts, below means they penalise interior streaks more.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    def runs(s):
        out, c = [], 1
        for x, y in zip(s, s[1:]):
            if x == y:
                c += 1
            else:
                out.append(c); c = 1
        out.append(c)
        return out
    R = {s: runs(s) for s in seqs}
    nsw = {s: len(r) for s, r in R.items()}
    mx = {s: max(r) for s, r in R.items()}
    edge = {s: float(max(r) in (r[0], r[-1])) for s, r in R.items()}
    a = df["sequence_a"]; b = df["sequence_b"]
    ea = a.map(edge).to_numpy(); eb = b.map(edge).to_numpy()
    m = ((a.map(nsw) == b.map(nsw)) & (a.map(mx) == b.map(mx))).to_numpy() & (ea != eb)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(ea[m] < eb[m], y, 1 - y)))
