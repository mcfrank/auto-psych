# name: partial_digit_majority_contrast
# description: In 3-4-example sets where all but exactly one example share a final digit, yes-rate for targets ending in that majority digit minus yes-rate for targets whose final digit matches no example; observed > null means people treat a near-unanimous digit as a rule (a near-coincidence the model ignores, since its rule/violation term needs all examples to agree), observed < null means the model over-credits the majority digit.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    S = s.to_numpy(dtype=float)
    valid = ~np.isnan(S)
    n = valid.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    D = np.where(valid, np.where(valid, S, 0) % 10, -1)
    counts = np.stack([(D == d).sum(axis=1) for d in range(10)], axis=1)
    maj = counts.argmax(axis=1)
    mc = counts.max(axis=1)
    part = (n >= 3) & (mc == n - 1)
    td = t % 10
    match_maj = td == maj
    match_any = (D == td[:, None]).any(axis=1)
    r = df["rating"].to_numpy(dtype=float)
    a = part & match_maj
    b = part & ~match_any
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
