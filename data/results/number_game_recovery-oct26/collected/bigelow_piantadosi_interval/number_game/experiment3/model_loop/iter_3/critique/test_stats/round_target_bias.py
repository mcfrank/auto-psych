# name: round_target_bias
# description: For sets whose examples are not all multiples of 10, yes rate on non-example targets that are multiples of 10 minus yes rate on other non-example targets; observed above null_mean means people favour round-number targets beyond what the model's concept posterior predicts (model lacks a round-number bias), below means the model over-endorses them.
def test_statistic(df):
    u = pd.Series(df["number_set"].unique())
    ex = {s: [int(x) for x in s.split(",")] for s in u}
    allround = df["number_set"].map({s: all(v % 10 == 0 for v in e) for s, e in ex.items()}).astype(bool)
    t = df["target"].astype(int)
    isex = pd.Series([tt in ex[s] for s, tt in zip(df["number_set"], t)], index=df.index)
    keep = (~allround) & (~isex)
    r = df["rating"].astype(float)
    rnd = keep & (t % 10 == 0)
    oth = keep & (t % 10 != 0)
    if rnd.sum() == 0 or oth.sum() == 0:
        return 0.0
    return float(r[rnd].mean() - r[oth].mean())
