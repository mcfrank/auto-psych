# name: multiple_of_five_set_contrast
# description: In sets with >=2 examples that are all multiples of 5, mean yes-rate for targets that are multiples of 5 minus mean yes-rate for targets that are not; observed > null means people extend a multiples-of-5 rule more sharply than the model's final-digit/magnitude similarity predicts, observed < null means the model over-generalizes along it.
def test_statistic(df):
    sets = df["number_set"].unique()
    ok = {}
    for s in sets:
        ex = [int(x) for x in s.split(",")]
        ok[s] = len(ex) >= 2 and all(e % 5 == 0 for e in ex)
    m = df["number_set"].map(ok).astype(bool)
    t = df["target"].astype(int)
    is5 = (t % 5 == 0)
    a = df.loc[m & is5, "rating"]
    b = df.loc[m & ~is5, "rating"]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
