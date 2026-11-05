# name: intermediate_cell_fraction
# description: Fraction of (set, target) cells whose mean yes rate lies in [0.25, 0.75]; observed above null means people's endorsements are more graded/uncertain than the model's posterior predicts, below null means more all-or-none.
def test_statistic(df):
    g = df.assign(rr=df["rating"].astype(float)).groupby(["number_set", "target"])["rr"].mean()
    return float(((g >= 0.25) & (g <= 0.75)).mean())
