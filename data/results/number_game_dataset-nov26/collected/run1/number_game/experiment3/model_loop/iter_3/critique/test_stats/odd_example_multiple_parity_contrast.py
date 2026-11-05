# name: odd_example_multiple_parity_contrast
# description: For one-example sets with an odd example e>1, yes-rate for odd multiples of e (t=k*e, k>1) minus yes-rate for even multiples of e; observed < null means the model's parity-switch cut on the multiples route is too strong (people accept even multiples like 14 after "7" nearly as often as odd ones), observed > null means the cut is too weak.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    e = pd.to_numeric(df["number_set"].str.split(",").str[0]).values
    t = df["target"].astype(int).values
    r = df["rating"].values
    base = (n.values == 1) & (e % 2 == 1) & (e > 1) & (t % e == 0) & (t != e)
    a = base & (t % 2 == 1)
    b = base & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
