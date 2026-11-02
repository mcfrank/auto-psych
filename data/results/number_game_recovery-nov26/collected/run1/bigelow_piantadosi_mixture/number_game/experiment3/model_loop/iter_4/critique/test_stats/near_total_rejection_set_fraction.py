# name: near_total_rejection_set_fraction
# description: Fraction of sets (with >=20 trials) whose overall yes rate is <=0.10; observed > null means people reject almost every target for some sets (all-or-none concept commitment) more often than the model's graded predictions produce, observed < null less.
def test_statistic(df):
    g = df.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) == 0:
        return 0.0
    return float((g["mean"] <= 0.10).mean())
