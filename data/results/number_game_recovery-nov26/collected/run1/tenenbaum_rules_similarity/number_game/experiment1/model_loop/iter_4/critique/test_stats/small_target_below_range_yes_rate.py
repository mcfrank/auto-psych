# name: small_target_below_range_yes_rate
# description: Yes-rate for non-example targets <= 10 that lie below the smallest example; observed < null means the model over-produces downward generalisation to small single-digit numbers (people rarely extend a set to the low end of 1-100).
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    sets = df["number_set"].astype(str).str.split(",").apply(lambda l: [int(x) for x in l])
    mn = sets.apply(min).to_numpy()
    m = (t <= 10) & (t < mn)
    r = df["rating"].to_numpy()
    return float(r[m].mean()) if m.any() else 0.0
