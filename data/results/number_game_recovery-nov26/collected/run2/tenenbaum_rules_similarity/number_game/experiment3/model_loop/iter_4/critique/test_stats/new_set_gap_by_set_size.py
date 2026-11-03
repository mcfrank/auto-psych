# name: new_set_gap_by_set_size
# description: (First-query minus repeat-query yes rate on non-example targets) for 1-2-example sets minus the same gap for 3-4-example sets; nonzero vs null means the new-set criterion shift depends on set size, which the model's single constant shift cannot capture (positive = small sets show a more liberal/less cautious first query relative to large sets).
def test_statistic(df):
    ns = df["number_set"].astype(str).values
    ti = df["trial_index"].values
    prev = np.concatenate([[""], ns[:-1]])
    first = (ns != prev) | (ti == 0)
    k = df["number_set"].astype(str).str.count(",").values + 1
    tgt = df["target"].astype(int).astype(str).values
    is_ex = np.array([t in s.split(",") for s, t in zip(ns, tgt)])
    r = df["rating"].values.astype(float)
    def gap(mask):
        a = mask & first & ~is_ex
        b = mask & ~first & ~is_ex
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return r[a].mean() - r[b].mean()
    return float(gap(k <= 2) - gap(k >= 3))
