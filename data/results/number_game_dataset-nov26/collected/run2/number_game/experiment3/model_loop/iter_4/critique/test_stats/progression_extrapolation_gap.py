# name: progression_extrapolation_gap
# description: On 2-4-example sets whose sorted examples form an arithmetic progression, among targets outside the examples' range, yes rate for targets continuing the step (e.g. 12 or 42 for 62,32,22) minus yes rate for other out-of-range targets; observed > null means people extrapolate the step pattern beyond the examples more sharply than the model (observed < null: the model over-extrapolates the progression).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    K = len(u)
    lo = np.zeros(K); hi = np.zeros(K); step = np.zeros(K, int); prog = np.zeros(K, bool)
    for i, s in enumerate(u):
        e = sorted(int(x) for x in s.split(","))
        lo[i], hi[i] = e[0], e[-1]
        if len(e) >= 2:
            g = e[1] - e[0]
            if g > 0 and all(e[j + 1] - e[j] == g for j in range(len(e) - 1)):
                prog[i] = True
                step[i] = g
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    st = np.maximum(step[inv], 1)
    hit = prog[inv] & ((t - lo[inv]) % st == 0) & (np.abs(t - lo[inv]) % st == 0)
    out = hit & ((t < lo[inv]) | (t > hi[inv]))
    outr = prog[inv] & ((t < lo[inv]) | (t > hi[inv]))
    inside = outr & ~hit
    if out.sum() == 0 or inside.sum() == 0:
        return 0.0
    return float(r[out].mean() - r[inside].mean())
