# name: single_example_near_target_yes
# description: For single-example sets, yes rate on non-example targets within 5 of the example; observed < null means the model over-generalises to numerically close numbers from one example (observed > null: under-generalises).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    single = ~sets.str.contains(",").to_numpy()
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ex = pd.to_numeric(sets.where(single, "-1000"), errors="coerce").fillna(-1000).astype(int).to_numpy()
    dist = np.abs(t - ex)
    m = single & (dist >= 1) & (dist <= 5)
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
