# name: between_set_mean_spread
# description: Standard deviation across sets (those asked with >=4 distinct targets) of the set-level mean yes-rate; observed > null means the model compresses set-to-set differences in overall willingness to say yes (e.g. near-zero for 84,94,34 vs high for 90,100,5), observed < null means it exaggerates them.
def test_statistic(df):
    g = df.groupby("number_set")
    nt = g["target"].nunique()
    m = g["rating"].mean()[nt >= 4]
    if len(m) < 2:
        return 0.0
    return float(m.std(ddof=0))
