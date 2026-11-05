# name: multi_multiple_of_max_gap
# description: On 2-4-example sets, yes rate for non-member targets that are multiples of the largest example (e.g. 64 for 8,32; 96 for 16,32,2,8) minus yes rate for other non-member targets; observed > null means people extend a scaling/multiplicative rule upward beyond what the model's shared-property profile predicts (observed < null: model over-extends).
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    u, inv = np.unique(sets, return_inverse=True)
    ex = [[int(x) for x in s.split(",")] for s in u]
    mx = np.array([max(e) for e in ex])
    nn = np.array([len(e) for e in ex])
    mem = np.zeros((len(u), 101), bool)
    for i, e in enumerate(ex):
        mem[i, e] = True
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    m = (nn[inv] >= 2) & ~mem[inv, t]
    hit = (t % mx[inv]) == 0
    a, b = m & hit, m & ~hit
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
