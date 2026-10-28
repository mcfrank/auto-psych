# name: exact_example_target_yes
# description: Yes-rate on trials whose target is one of the shown examples, minus yes-rate on targets within 1-3 of the nearest example; observed > null means people treat a repeated example as near-certainly produced far more than nearby numbers, and the model under-produces this boost.
def test_statistic(df):
    g = df.groupby(["number_set", "target"], sort=False)["rating"].mean().reset_index()
    hit, near = [], []
    for s, t, r in zip(g["number_set"], g["target"].astype(int), g["rating"]):
        ex = [int(x) for x in str(s).split(",")]
        d = min(abs(e - t) for e in ex)
        if d == 0:
            hit.append(r)
        elif d <= 3:
            near.append(r)
    if not hit or not near:
        return 0.0
    return float(np.mean(hit) - np.mean(near))
