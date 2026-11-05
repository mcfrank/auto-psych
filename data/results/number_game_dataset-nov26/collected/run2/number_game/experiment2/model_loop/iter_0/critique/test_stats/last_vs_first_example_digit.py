# name: last_vs_first_example_digit
# description: In multi-example sets whose first and last shown examples have different final digits, yes-rate for targets sharing the LAST example's final digit minus yes-rate for targets sharing the FIRST example's final digit; nonzero observed vs null means presentation order (recency/primacy) matters, which the order-blind model cannot produce (positive = recency).
def test_statistic(df):
    fl = {}
    for s in df["number_set"].unique():
        ex = [int(x) for x in str(s).split(",")]
        if len(ex) >= 2 and ex[0] % 10 != ex[-1] % 10:
            fl[s] = (ex[0] % 10, ex[-1] % 10)
    sub = df[df["number_set"].isin(list(fl.keys()))]
    if len(sub) == 0:
        return 0.0
    f = sub["number_set"].map(lambda s: fl[s][0]).values
    l = sub["number_set"].map(lambda s: fl[s][1]).values
    td = sub["target"].astype(int).values % 10
    r = sub["rating"].values.astype(float)
    hl = td == l
    hf = td == f
    if hl.sum() == 0 or hf.sum() == 0:
        return 0.0
    return float(r[hl].mean() - r[hf].mean())
