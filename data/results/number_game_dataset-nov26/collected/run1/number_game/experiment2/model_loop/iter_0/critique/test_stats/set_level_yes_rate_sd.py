# name: set_level_yes_rate_sd
# description: Standard deviation across sets (with >=3 distinct targets) of each set's mean yes-rate; observed > null means overall willingness to say yes varies between sets more than the model's single bias/slope mapping from summed similarity produces (set-specific rule breadth), observed < null means less.
def test_statistic(df):
    g = df.groupby("number_set").agg(m=("rating", "mean"), k=("target", "nunique"))
    g = g[g["k"] >= 3]
    return float(g["m"].std()) if len(g) > 1 else 0.0
