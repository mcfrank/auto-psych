# name: n4_set_mean_variance
# description: Variance across four-example sets of each set's mean yes rate; observed > null means people's breadth of generalization differs more between four-example sets than the memory-drop model predicts (observed < null: the model over-differentiates four-example sets).
def test_statistic(df):
    s = df["number_set"].astype(str)
    n = s.str.count(",").values + 1
    m = n == 4
    if m.sum() == 0:
        return 0.0
    g = df.loc[m, "rating"].astype(float).groupby(s[m].values).mean()
    return float(g.var(ddof=0))
