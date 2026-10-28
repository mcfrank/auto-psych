# name: first_vs_last_example_proximity
# description: In sets of >=2 examples, among non-example targets within 1-10 of their nearest example, yes-rate when that nearest example was shown first minus when it was shown last; observed > null means people weight the first example (primacy) more than the model's recency-weighted memory allows, observed < null means recency is stronger than modelled.
def test_statistic(df):
    g = df.groupby(["number_set", "target"], sort=False)["rating"].mean().reset_index()
    first, last = [], []
    for s, t, r in zip(g["number_set"], g["target"].astype(int), g["rating"]):
        ex = [int(x) for x in str(s).split(",")]
        if len(ex) < 2 or t in ex:
            continue
        d = [abs(e - t) for e in ex]
        i = int(np.argmin(d))
        if d[i] > 10:
            continue
        if i == 0:
            first.append(r)
        elif i == len(ex) - 1:
            last.append(r)
    if not first or not last:
        return 0.0
    return float(np.mean(first) - np.mean(last))
