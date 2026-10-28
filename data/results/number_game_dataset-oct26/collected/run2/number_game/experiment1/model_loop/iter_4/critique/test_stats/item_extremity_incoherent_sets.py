# name: item_extremity_incoherent_sets
# description: Among (set, target) items from single-example or low-coherence sets (mean pairwise coherence over size/digit/parity/shared-factor < 0.4), the fraction of items whose yes rate is <= 1/9 or >= 8/9; observed > model means people answer decisively even after incoherent sets, contrary to the model's baseline-noisy decision rule there (observed < model means they are noisier than predicted).
def test_statistic(df):
    s = df["number_set"].astype(str)
    cmap = {}
    for u in s.unique():
        v = [int(z) for z in u.split(",")]
        pr = [(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
        if not pr:
            cmap[u] = 0.0
            continue
        def fm(a, b):
            m = min(a, b)
            return 0.0 if m <= 1 else 1.0 - math.log(math.gcd(a, b)) / math.log(m)
        c0 = np.mean([max(0.0, 1 - abs(a - b) / 50.0) for a, b in pr])
        c1 = np.mean([a % 10 == b % 10 for a, b in pr])
        c2 = np.mean([a % 2 == b % 2 for a, b in pr])
        c3 = np.mean([1 - fm(a, b) for a, b in pr])
        cmap[u] = float((c0 + c1 + c2 + c3) / 4.0)
    coh = s.map(cmap).values
    d = pd.DataFrame({"s": s.values, "t": df["target"].astype(int).values,
                      "y": df["rating"].astype(float).values})
    d = d[coh < 0.4]
    if len(d) == 0:
        return 0.0
    m = d.groupby(["s", "t"])["y"].mean().values
    return float(np.mean((m <= 1.0 / 9 + 1e-9) | (m >= 8.0 / 9 - 1e-9)))
