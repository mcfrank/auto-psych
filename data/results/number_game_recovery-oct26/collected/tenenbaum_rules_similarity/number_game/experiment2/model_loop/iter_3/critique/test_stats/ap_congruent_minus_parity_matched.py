# name: ap_congruent_minus_parity_matched
# description: For sets of >=2 examples whose pairwise differences share a gcd g in 3..9 with nonzero offset (an arithmetic progression not captured by any multiple-of rule, e.g. 83,77 or 53,47,41), the yes-rate on non-example targets congruent to the examples mod g minus the yes-rate on same-parity non-congruent non-example targets within the same distance (<=3g of the nearest example); observed > null means people extrapolate step-progressions more than the model's rule space allows.
def test_statistic(df):
    sets = pd.Series(df["number_set"].unique())
    info = {}
    for s in sets:
        ex = [int(x) for x in s.split(",")]
        if len(ex) < 2:
            continue
        g = 0
        for x in ex[1:]:
            g = math.gcd(g, abs(x - ex[0]))
        if 3 <= g <= 9 and ex[0] % g != 0:
            info[s] = (g, ex[0] % g, ex[0] % 2 if len(set(e % 2 for e in ex)) == 1 else -1, ex)
    if not info:
        return 0.0
    sub = df[df["number_set"].isin(list(info.keys()))]
    t = sub["target"].astype(int).to_numpy()
    ns = sub["number_set"].to_numpy()
    g = np.array([info[s][0] for s in ns])
    off = np.array([info[s][1] for s in ns])
    par = np.array([info[s][2] for s in ns])
    exl = [info[s][3] for s in ns]
    dist = np.array([min(abs(tt - e) for e in ex) for tt, ex in zip(t, exl)])
    nonex = dist > 0
    near = dist <= 3 * g
    cong = (t % g) == off
    samepar = (par < 0) | ((t % 2) == par)
    r = sub["rating"].to_numpy().astype(float)
    a = nonex & near & cong
    b = nonex & near & ~cong & samepar
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
