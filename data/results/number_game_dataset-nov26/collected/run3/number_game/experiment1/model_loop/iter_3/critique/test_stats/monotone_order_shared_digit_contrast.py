# name: monotone_order_shared_digit_contrast
# description: Among 3-example sets whose examples all share a last digit, yes rate of non-example targets lacking that digit for sets shown in monotone (sorted ascending/descending) order minus sets shown in non-monotone order; observed above null means presentation order (an orderly sequence) loosens the "ends in d" rule, which the order-blind model cannot produce (below null means the reverse).

def test_statistic(df):
    d = df[["number_set", "target", "rating"]].copy()
    d["target"] = d["target"].astype(int)
    mono, non = [], []
    for s, g in d.groupby("number_set"):
        ex = [int(x) for x in s.split(",")]
        if len(ex) != 3 or len({e % 10 for e in ex}) != 1:
            continue
        g = g[(g["target"] % 10) != ex[0] % 10]
        if len(g) == 0:
            continue
        is_mono = ex == sorted(ex) or ex == sorted(ex, reverse=True)
        (mono if is_mono else non).append(g["rating"].to_numpy())
    if not mono or not non:
        return 0.0
    return float(np.concatenate(mono).mean() - np.concatenate(non).mean())
