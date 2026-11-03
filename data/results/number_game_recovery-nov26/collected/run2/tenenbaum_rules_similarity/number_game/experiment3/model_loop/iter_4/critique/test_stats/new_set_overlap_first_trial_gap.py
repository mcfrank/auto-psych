# name: new_set_overlap_first_trial_gap
# description: On the first trial after a set change (non-example targets, trial_index > 0), yes rate when the new set shares at least one example with the previous trial's set minus when it shares none; observed above null means overlapping sets are treated as continuations (less new-set caution or carried-over belief), which the model's uniform first-trial shift misses.
def test_statistic(df):
    ns = df["number_set"].astype(str).values
    ti = df["trial_index"].values
    prev = np.concatenate([[""], ns[:-1]])
    first = (ns != prev) & (ti > 0)
    tgt = df["target"].astype(int).astype(str).values
    is_ex = np.array([t in s.split(",") for s, t in zip(ns, tgt)])
    ov = np.array([bool(set(a.split(",")) & set(b.split(","))) for a, b in zip(ns, prev)])
    r = df["rating"].values.astype(float)
    a = first & ov & ~is_ex
    b = first & ~ov & ~is_ex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
