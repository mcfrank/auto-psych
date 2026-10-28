# name: cell_extremity
# description: Mean over (set,target) cells of |cell yes-rate - 0.5|; observed below null means the model's predictions are over-confident (too close to 0/1) relative to graded human responding.
def test_statistic(df):
    m = df.groupby(["number_set", "target"])["rating"].mean()
    return float((m - 0.5).abs().mean())
