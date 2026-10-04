# name: interior_longest_run_choice
# description: Among pairs with equal switch counts and equal longest-run length where exactly one sequence has a longest run touching an end (first or last flip), the rate of choosing the sequence whose longest run is interior; observed above null means people find edge streaks less random than the model (blind to streak position) predicts, below means interior streaks look less random.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    seqs = pd.unique(pd.concat([a, b]))
    def info(s):
        bounds, start = [], 0
        for i in range(1, len(s) + 1):
            if i == len(s) or s[i] != s[i - 1]:
                bounds.append((start, i)); start = i
        L = max(e - st for st, e in bounds)
        edge = any((e - st) == L and (st == 0 or e == len(s)) for st, e in bounds)
        return (len(bounds), L, int(edge))
    inf = {s: info(s) for s in seqs}
    na = a.map(lambda s: inf[s][0]); nb = b.map(lambda s: inf[s][0])
    la = a.map(lambda s: inf[s][1]); lb = b.map(lambda s: inf[s][1])
    ea = a.map(lambda s: inf[s][2]); eb = b.map(lambda s: inf[s][2])
    m = (na == nb) & (la == lb) & (ea != eb)
    if m.sum() == 0:
        return 0.5
    ch = np.where(ea[m] == 0, df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(ch))
