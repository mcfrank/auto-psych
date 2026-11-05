# name: partial_last_digit_majority_contrast
# description: For sets where some last digit is shared by >=2 but not all examples, mean over sets of (yes rate of non-example targets ending in that shared digit minus yes rate of other non-example targets); observed above null means people generalize from a feature most (not all) examples share, which the model's all-or-nothing rule consistency under-produces.

def test_statistic(df):
    d = df[["number_set", "target", "rating"]].copy()
    d["target"] = d["target"].astype(int)
    diffs = []
    for s, g in d.groupby("number_set"):
        ex = [int(x) for x in s.split(",")]
        if len(ex) < 3:
            continue
        ld = [e % 10 for e in ex]
        shared = {x for x in ld if ld.count(x) >= 2 and ld.count(x) < len(ex)}
        if not shared:
            continue
        g = g[~g["target"].isin(ex)]
        m = (g["target"] % 10).isin(list(shared))
        if m.sum() == 0 or (~m).sum() == 0:
            continue
        diffs.append(g.loc[m, "rating"].mean() - g.loc[~m, "rating"].mean())
    return float(np.mean(diffs)) if diffs else 0.0
