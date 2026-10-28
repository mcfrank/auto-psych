# name: cell_rate_dispersion
# description: Variance across (set, target) cells of the cell's yes-rate (each cell has repeated answers); observed above null means the model's per-cell predictions are too flat/undifferentiated (people discriminate cells more sharply), below means the model is too extreme/deterministic across cells.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(int).astype(str)
    m = df["rating"].astype(float).groupby(key).mean()
    if len(m) < 2:
        return 0.0
    return float(m.var())
