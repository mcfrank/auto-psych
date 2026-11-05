# name: power_of_two_contrast
# description: In sets whose examples are all powers of 2, mean yes-rate for power-of-2 targets minus mean yes-rate for other targets (difference of two trial-level means; 0 if a group is empty); observed > null means people use a powers-of-2 rule more sharply than the model's magnitude/digit/parity/shared-factor similarity predicts, observed < null means less.
def test_statistic(df):
    p2 = {1, 2, 4, 8, 16, 32, 64}
    sets = df["number_set"].unique()
    ok = {s: all(int(x) in p2 for x in s.split(",")) for s in sets}
    m = df["number_set"].map(ok).astype(bool)
    t = df["target"].astype(int)
    isp = t.isin(list(p2))
    a = df.loc[m & isp, "rating"]
    b = df.loc[m & ~isp, "rating"]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
