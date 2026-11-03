# name: four_minus_two_examples_outside
# description: Yes-rate for targets outside the examples' range in 4-example sets minus that in 2-example sets; observed more negative than null means people tighten generalisation with more examples faster than the model's size principle predicts.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = s.str.count(",").to_numpy() + 1
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    out = (t < np.nanmin(ex, axis=1)) | (t > np.nanmax(ex, axis=1))
    m4 = out & (n == 4); m2 = out & (n == 2)
    a = r[m4].mean() if m4.any() else 0.0; b = r[m2].mean() if m2.any() else 0.0
    return float(a - b)
