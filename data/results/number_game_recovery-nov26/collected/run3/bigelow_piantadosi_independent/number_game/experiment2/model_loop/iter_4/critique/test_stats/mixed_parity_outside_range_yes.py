# name: mixed_parity_outside_range_yes
# description: In sets of >=3 examples with both odd and even members, yes rate on targets outside [min,max] of the examples; observed > null means people guess/extend broadly when no rule fits more than the model's interval fallback and lapse predict.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    feat = {}
    for s in sets.unique():
        ex = [int(v) for v in s.split(",")]
        mixed = len(ex) >= 3 and len({v % 2 for v in ex}) == 2
        feat[s] = (mixed, min(ex), max(ex))
    ok = sets.map(lambda s: feat[s][0]).to_numpy(bool)
    lo = sets.map(lambda s: feat[s][1]).to_numpy(int)
    hi = sets.map(lambda s: feat[s][2]).to_numpy(int)
    m = ok & ((t < lo) | (t > hi))
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
