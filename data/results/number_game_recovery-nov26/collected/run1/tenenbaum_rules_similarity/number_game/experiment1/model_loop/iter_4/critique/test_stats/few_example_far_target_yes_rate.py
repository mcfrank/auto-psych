# name: few_example_far_target_yes_rate
# description: For sets of 2-3 examples, yes-rate on targets more than 20 outside [min,max] of the examples (excluding targets <=10 and target 100); observed < null means the model over-generalises far beyond the examples' range from few examples (e.g. 28,2 -> 64), i.e. people's generalisation is more magnitude-local than its broad rules imply.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    sets = df["number_set"].astype(str).str.split(",").apply(lambda l: [int(x) for x in l])
    k = sets.str.len().to_numpy()
    mn = sets.apply(min).to_numpy(); mx = sets.apply(max).to_numpy()
    dist = np.maximum(0, np.maximum(mn - t, t - mx))
    m = (k >= 2) & (k <= 3) & (dist > 20) & (t > 10) & (t < 100)
    r = df["rating"].to_numpy()
    return float(r[m].mean()) if m.any() else 0.0
